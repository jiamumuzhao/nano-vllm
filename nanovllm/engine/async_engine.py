import asyncio
from dataclasses import dataclass, field
from time import time
from typing import AsyncIterator

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    text: str
    token_ids: list[int]
    delta_text: str
    delta_token_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None


@dataclass(slots=True)
class _RequestState:
    request_id: str
    queue: asyncio.Queue[RequestOutput | None] = field(default_factory=asyncio.Queue)
    seq_id: int | None = None
    token_ids: list[int] = field(default_factory=list)
    text: str = ""


class AsyncEngine:
    """Async serving engine with continuous batching and streaming outputs."""

    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(model, **kwargs)
        self._new_requests: asyncio.Queue[tuple[str | list[int], SamplingParams, _RequestState]] = asyncio.Queue()
        self._streams: dict[int, _RequestState] = {}
        self._closed = False
        self._runner_task: asyncio.Task | None = None

    async def start(self):
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._run_loop())

    async def shutdown(self):
        self._closed = True
        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
        self.engine.exit()

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        if self._closed:
            raise RuntimeError("AsyncEngine is closed")
        await self.start()
        state = _RequestState(request_id=request_id or f"cmpl-{int(time() * 1_000_000)}")
        await self._new_requests.put((prompt, sampling_params, state))
        while True:
            item = await state.queue.get()
            if item is None:
                break
            yield item

    async def _run_loop(self):
        while not self._closed:
            await self._drain_new_requests()
            if self.engine.is_finished():
                await asyncio.sleep(0.001)
                continue

            _, _, token_events = self.engine.step_with_events()
            for seq_id, token_id, finished in token_events:
                state = self._streams.get(seq_id)
                if state is None:
                    continue
                state.token_ids.append(token_id)
                text = self.engine.tokenizer.decode(state.token_ids)
                delta_text = text[len(state.text):]
                state.text = text
                output = RequestOutput(
                    request_id=state.request_id,
                    text=state.text,
                    token_ids=list(state.token_ids),
                    delta_text=delta_text,
                    delta_token_ids=[token_id],
                    finished=finished,
                    finish_reason="stop" if finished else None,
                )
                await state.queue.put(output)
                if finished:
                    self._streams.pop(seq_id, None)
                    await state.queue.put(None)
            await asyncio.sleep(0)

    async def _drain_new_requests(self):
        while True:
            try:
                prompt, sampling_params, state = self._new_requests.get_nowait()
            except asyncio.QueueEmpty:
                return
            seq_id = self.engine.add_request(prompt, sampling_params)
            state.seq_id = seq_id
            self._streams[seq_id] = state
