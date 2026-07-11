import argparse
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.async_engine import AsyncEngine
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

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(app.state.engine, prompt, sampling_params, model_name, created),
                media_type="text/event-stream",
            )

        final = None
        async for output in app.state.engine.generate(prompt, sampling_params):
            final = output
        text = final.text if final else ""
        token_ids = final.token_ids if final else []
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(token_ids)
        return {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
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

        if request.stream:
            return StreamingResponse(
                _stream_completion(app.state.engine, prompts[0], sampling_params, model_name, created),
                media_type="text/event-stream",
            )

        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        tokenizer = app.state.engine.engine.tokenizer
        for index, prompt in enumerate(prompts):
            final = None
            async for output in app.state.engine.generate(prompt, sampling_params):
                final = output
            text = final.text if final else ""
            token_ids = final.token_ids if final else []
            choices.append({
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": "stop",
            })
            prompt_tokens += len(tokenizer.encode(prompt))
            completion_tokens += len(token_ids)

        return {
            "id": f"cmpl-{created}",
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
):
    request_id = f"cmpl-{created}"
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--enforce-eager", action="store_true")
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
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
    )
    app = create_app(engine, args.served_model_name or args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
