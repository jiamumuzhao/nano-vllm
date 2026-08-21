import argparse
import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.async_engine import AsyncEngine, QueueFullError
from nanovllm.sampling_params import SamplingParams


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, gt=1e-10)
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, gt=1e-10)
    stream: bool = False
    ignore_eos: bool = False


def create_app(engine: AsyncEngine, served_model_name: str) -> FastAPI:
    app = FastAPI(title="Nano-vLLM OpenAI-compatible API")
    app.state.engine = engine
    app.state.served_model_name = served_model_name

    @app.on_event("startup")
    async def startup():
        await app.state.engine.start()

    @app.on_event("shutdown")
    async def shutdown():
        await app.state.engine.shutdown()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": app.state.served_model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nano-vllm",
            }],
        }

    @app.get("/v1/requests/{request_id}")
    async def request_status(request_id: str):
        status = app.state.engine.get_request_status(request_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"unknown request_id: {request_id}")
        return status

    @app.delete("/v1/requests/{request_id}")
    async def cancel_request(request_id: str):
        status = await app.state.engine.cancel(request_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"unknown request_id: {request_id}")
        return status

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        tokenizer = app.state.engine.engine.tokenizer
        messages = [message.model_dump() for message in request.messages]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        model_name = request.model or app.state.served_model_name
        created = int(time.time())
        request_id = AsyncEngine.new_request_id("chatcmpl")
        if getattr(app.state.engine, "is_unavailable", lambda: False)():
            detail = getattr(app.state.engine, "engine_error", None) or "engine is unavailable"
            raise HTTPException(status_code=503, detail=detail)
        if not app.state.engine.can_accept_request():
            raise HTTPException(status_code=429, detail="request queue is full")

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(app.state.engine, prompt, sampling_params, model_name, created, request_id),
                media_type="text/event-stream",
            )

        final = None
        try:
            async for output in app.state.engine.generate(prompt, sampling_params, request_id=request_id):
                final = output
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"request failed: {exc}") from exc
        finally:
            status = app.state.engine.get_request_status(request_id)
            if status and status["status"] not in {"finished", "cancelled", "failed"}:
                await app.state.engine.cancel(request_id)
        text = final.text if final else ""
        token_ids = final.token_ids if final else []
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(token_ids)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": final.finish_reason if final else "error",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        if request.stream and len(prompts) != 1:
            raise HTTPException(status_code=400, detail="streaming currently supports exactly one prompt")

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        model_name = request.model or app.state.served_model_name
        created = int(time.time())
        request_ids = [AsyncEngine.new_request_id("cmpl") for _ in prompts]
        if getattr(app.state.engine, "is_unavailable", lambda: False)():
            detail = getattr(app.state.engine, "engine_error", None) or "engine is unavailable"
            raise HTTPException(status_code=503, detail=detail)
        if not app.state.engine.can_accept_request(len(prompts)):
            raise HTTPException(status_code=429, detail="request queue is full")

        if request.stream:
            return StreamingResponse(
                _stream_completion(app.state.engine, prompts[0], sampling_params, model_name, created, request_ids[0]),
                media_type="text/event-stream",
            )

        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        tokenizer = app.state.engine.engine.tokenizer
        finals = []
        try:
            for index, prompt in enumerate(prompts):
                final = None
                async for output in app.state.engine.generate(prompt, sampling_params, request_id=request_ids[index]):
                    final = output
                finals.append(final)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"request failed: {exc}") from exc
        finally:
            for request_id in request_ids:
                status = app.state.engine.get_request_status(request_id)
                if status and status["status"] not in {"finished", "cancelled", "failed"}:
                    await app.state.engine.cancel(request_id)
        for index, prompt in enumerate(prompts):
            final = finals[index]
            text = final.text if final else ""
            token_ids = final.token_ids if final else []
            choices.append({
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": final.finish_reason if final else "error",
            })
            prompt_tokens += len(tokenizer.encode(prompt))
            completion_tokens += len(token_ids)

        return {
            "id": request_ids[0],
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


async def _stream_completion(
    engine: AsyncEngine,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
    created: int,
    request_id: str,
):
    try:
        async for output in engine.generate(prompt, sampling_params, request_id=request_id):
            chunk: dict[str, Any] = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": output.delta_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": output.finish_reason if output.finished else None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        status = engine.get_request_status(request_id)
        if status and status["status"] not in {"finished", "cancelled", "failed"}:
            await engine.cancel(request_id)


async def _stream_chat_completion(
    engine: AsyncEngine,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
    created: int,
    request_id: str,
):
    try:
        async for output in engine.generate(prompt, sampling_params, request_id=request_id):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": output.delta_text},
                    "finish_reason": output.finish_reason if output.finished else None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        status = engine.get_request_status(request_id)
        if status and status["status"] not in {"finished", "cancelled", "failed"}:
            await engine.cancel(request_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve Nano-vLLM with an OpenAI-compatible completions API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--prefix-cache-max-blocks", type=int, default=-1, help="maximum inactive prefix-cache blocks; -1 means all KV blocks")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-queue-size", type=int, default=256)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--stream-queue-size", type=int, default=16)
    return parser.parse_args()


def main():
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install serving dependencies with `pip install -e .[serve]`.") from exc

    args = parse_args()
    engine = AsyncEngine(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefix_cache_max_blocks=args.prefix_cache_max_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        max_queue_size=args.max_queue_size,
        request_timeout_s=args.request_timeout_s,
        stream_queue_size=args.stream_queue_size,
    )
    app = create_app(engine, args.served_model_name or args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
