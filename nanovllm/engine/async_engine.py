"""Async serving lifecycle, cancellation and bounded backpressure."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from time import monotonic
from typing import AsyncIterator

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams


REQUEST_STATUSES = {"queued", "prefill", "decode", "finished", "cancelled", "failed"}
TERMINAL_STATUSES = {"finished", "cancelled", "failed"}


class QueueFullError(RuntimeError):
    """The configured intake queue is full; API callers map this to HTTP 429."""

    status_code = 429


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    text: str
    token_ids: list[int]
    delta_text: str
    delta_token_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None
    status: str = "decode"


@dataclass(slots=True)
class _RequestState:
    request_id: str
    queue: asyncio.Queue[RequestOutput]
    seq_id: int | None = None
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    status: str = "queued"
    finish_reason: str | None = None
    deadline: float = 0.0
    error: str | None = None
    created_at: float = 0.0
    first_token_at: float | None = None
    completed_at: float | None = None
    prompt_tokens: int = 0
    metrics_recorded: bool = False
    terminal_emitted: bool = False


class AsyncEngine:
    """Async engine with explicit request lifecycle and bounded queues."""

    def __init__(self, model: str, **kwargs):
        self.max_queue_size = self._positive_int(kwargs.pop("max_queue_size", 256), "max_queue_size")
        self.request_timeout_s = self._positive_float(kwargs.pop("request_timeout_s", 300.0), "request_timeout_s")
        self.stream_queue_size = self._positive_int(kwargs.pop("stream_queue_size", 16), "stream_queue_size")
        self.engine = LLMEngine(model, **kwargs)
        self._new_requests: asyncio.Queue[tuple[str | list[int], SamplingParams, _RequestState]] = asyncio.Queue()
        self._states: dict[str, _RequestState] = {}
        self._streams: dict[int, _RequestState] = {}
        self._seq_to_request: dict[int, str] = {}
        self._closed = False
        self._engine_error: str | None = None
        self._metric_counters = {
            "requests_total": 0,
            "requests_finished_total": 0,
            "requests_cancelled_total": 0,
            "requests_failed_total": 0,
            "prompt_tokens_total": 0,
            "generation_tokens_total": 0,
            "first_tokens_total": 0,
        }
        self._metric_sums = {
            "ttft_seconds": 0.0,
            "e2e_request_seconds": 0.0,
        }
        self._metric_counts = {
            "ttft_seconds": 0,
            "e2e_request_seconds": 0,
        }
        self._runner_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _positive_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _positive_float(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        return float(value)

    @staticmethod
    def new_request_id(prefix: str = "req") -> str:
        return f"{prefix}-{uuid.uuid4()}"

    async def start(self):
        if self._closed:
            raise RuntimeError("AsyncEngine is closed")
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._run_loop())

    def _active_count(self) -> int:
        return sum(state.status not in TERMINAL_STATUSES for state in self._states.values())

    def can_accept_request(self, count: int = 1) -> bool:
        return not self._closed and isinstance(count, int) and count > 0 and self._active_count() + count <= self.max_queue_size

    def is_unavailable(self) -> bool:
        return self._engine_error is not None

    @property
    def engine_error(self) -> str | None:
        return self._engine_error

    def get_request_status(self, request_id: str):
        state = self._states.get(request_id)
        if state is None:
            return None
        return {
            "request_id": state.request_id,
            "status": state.status,
            "finish_reason": state.finish_reason,
            "seq_id": state.seq_id,
            "error": state.error,
        }

    def _record_first_token(self, state: _RequestState):
        if state.first_token_at is not None:
            return
        state.first_token_at = monotonic()
        self._metric_counters["first_tokens_total"] += 1
        ttft = state.first_token_at - state.created_at
        if math.isfinite(ttft) and ttft >= 0:
            self._metric_sums["ttft_seconds"] += ttft
            self._metric_counts["ttft_seconds"] += 1

    def _record_terminal_metrics(self, state: _RequestState):
        if state.metrics_recorded:
            return
        state.metrics_recorded = True
        state.completed_at = monotonic()
        self._metric_counters["prompt_tokens_total"] += state.prompt_tokens
        self._metric_counters["generation_tokens_total"] += len(state.token_ids)
        if state.status == "finished":
            self._metric_counters["requests_finished_total"] += 1
        elif state.status == "cancelled":
            self._metric_counters["requests_cancelled_total"] += 1
        elif state.status == "failed":
            self._metric_counters["requests_failed_total"] += 1
        e2e = state.completed_at - state.created_at
        if math.isfinite(e2e) and e2e >= 0:
            self._metric_sums["e2e_request_seconds"] += e2e
            self._metric_counts["e2e_request_seconds"] += 1

    def get_metrics_snapshot(self) -> dict:
        """Return read-only service metrics for external observers/benchmarks."""
        scheduler = getattr(self.engine, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "get_metrics_snapshot"):
            scheduler_metrics = dict(scheduler.get_metrics_snapshot())
        else:
            block_manager = getattr(scheduler, "block_manager", None)
            used = len(getattr(block_manager, "used_block_ids", ())) if block_manager is not None else 0
            total = len(getattr(block_manager, "blocks", ())) if block_manager is not None else 0
            scheduler_metrics = {
                "kv_blocks_total": total,
                "kv_blocks_used": used,
                "kv_blocks_peak_used": used,
                "kv_usage_peak_ratio": used / total if total else 0.0,
                "preemption_count": int(getattr(scheduler, "preemption_count", 0)) if scheduler is not None else 0,
                "prefix_cache_requests": 0,
                "prefix_cache_hit_requests": 0,
                "prefix_cache_cached_tokens": 0,
                "prefix_cache_hit_rate": 0.0,
                "prefix_cache_token_hit_rate": 0.0,
            }
        counts = {status: 0 for status in REQUEST_STATUSES}
        for state in self._states.values():
            if state.status in counts:
                counts[state.status] += 1
        metric_counters = getattr(self, "_metric_counters", {})
        metric_sums = getattr(self, "_metric_sums", {})
        metric_counts = getattr(self, "_metric_counts", {})
        return {
            "scheduler": scheduler_metrics,
            "active_requests": sum(counts[s] for s in REQUEST_STATUSES - TERMINAL_STATUSES),
            "queued_requests": counts["queued"],
            "prefill_requests": counts["prefill"],
            "decode_requests": counts["decode"],
            "finished_requests": counts["finished"],
            "cancelled_requests": counts["cancelled"],
            "failed_requests": counts["failed"],
            "intake_queue_length": self._new_requests.qsize(),
            "max_queue_size": self.max_queue_size,
            "stream_queue_size": self.stream_queue_size,
            "request_timeout_s": self.request_timeout_s,
            "engine_unavailable": int(getattr(self, "_engine_error", None) is not None),
            "requests_total": metric_counters.get("requests_total", 0),
            "requests_finished_total": metric_counters.get("requests_finished_total", 0),
            "requests_cancelled_total": metric_counters.get("requests_cancelled_total", 0),
            "requests_failed_total": metric_counters.get("requests_failed_total", 0),
            "prompt_tokens_total": metric_counters.get("prompt_tokens_total", 0),
            "generation_tokens_total": metric_counters.get("generation_tokens_total", 0),
            "first_tokens_total": metric_counters.get("first_tokens_total", 0),
            "ttft_seconds_sum": metric_sums.get("ttft_seconds", 0.0),
            "ttft_seconds_count": metric_counts.get("ttft_seconds", 0),
            "e2e_request_seconds_sum": metric_sums.get("e2e_request_seconds", 0.0),
            "e2e_request_seconds_count": metric_counts.get("e2e_request_seconds", 0),
        }

    async def shutdown(self):
        if self._closed and self._runner_task is None:
            return
        self._closed = True
        for request_id in list(self._states):
            await self.cancel(request_id, reason="cancelled")
        if self._runner_task is not None:
            task, self._runner_task = self._runner_task, None
            if not task.done():
                task.cancel()
            try:
                await task
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
            detail = f"; {self._engine_error}" if self._engine_error else ""
            raise RuntimeError(f"AsyncEngine is closed{detail}")
        await self.start()
        request_id = request_id or self.new_request_id("req")
        async with self._lock:
            if request_id in self._states:
                raise ValueError(f"duplicate request_id: {request_id}")
            if self._active_count() >= self.max_queue_size:
                raise QueueFullError(f"request queue is full (max_queue_size={self.max_queue_size})")
            state = _RequestState(
                request_id=request_id,
                queue=asyncio.Queue(maxsize=self.stream_queue_size),
                created_at=monotonic(),
                deadline=monotonic() + self.request_timeout_s,
            )
            self._metric_counters["requests_total"] += 1
            self._states[request_id] = state
            await self._new_requests.put((prompt, sampling_params, state))
        try:
            while True:
                item = await state.queue.get()
                yield item
                # A terminal RequestOutput is the sole end-of-stream marker.
                # No sentinel task is needed, so a disconnected consumer can
                # never leave a blocked producer task behind.
                if item.finished:
                    break
        finally:
            if state.status not in TERMINAL_STATUSES:
                await self.cancel(request_id, reason="cancelled")

    def _sequence_status(self, seq) -> str:
        mapping = {
            SequenceStatus.QUEUED: "queued",
            SequenceStatus.PREFILL: "prefill",
            SequenceStatus.DECODE: "decode",
            SequenceStatus.FINISHED: "finished",
            SequenceStatus.CANCELLED: "cancelled",
            SequenceStatus.FAILED: "failed",
        }
        return mapping.get(seq.status, "decode")

    def _sync_sequence_statuses(self):
        scheduler = getattr(self.engine, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "get_sequence"):
            return
        for seq_id, request_id in list(self._seq_to_request.items()):
            state = self._states.get(request_id)
            seq = scheduler.get_sequence(seq_id)
            if state is not None and seq is not None and state.status not in TERMINAL_STATUSES:
                mapped = self._sequence_status(seq)
                # The scheduler marks a sequence FINISHED while postprocess is
                # completing the step.  The token event still owns publication
                # of the final RequestOutput (including the stop reason), so do
                # not let this bookkeeping update hide that event.
                if mapped != "finished":
                    state.status = mapped

    def _clear_queue(self, state: _RequestState):
        while True:
            try:
                state.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _terminal_output(self, state: _RequestState) -> RequestOutput:
        return RequestOutput(
            request_id=state.request_id,
            text=state.text,
            token_ids=list(state.token_ids),
            delta_text="",
            delta_token_ids=[],
            finished=True,
            finish_reason=state.finish_reason,
            status=state.status,
        )

    def _enqueue_terminal(self, state: _RequestState):
        if state.terminal_emitted:
            return
        self._clear_queue(state)
        state.queue.put_nowait(self._terminal_output(state))
        state.terminal_emitted = True

    def _complete(self, state: _RequestState, status: str, reason: str, error: str | None = None):
        if state.status in TERMINAL_STATUSES:
            return
        state.status = status
        state.finish_reason = reason
        state.error = error
        self._record_terminal_metrics(state)
        if state.seq_id is not None:
            self._streams.pop(state.seq_id, None)
            self._seq_to_request.pop(state.seq_id, None)
            scheduler = getattr(self.engine, "scheduler", None)
            if scheduler is not None:
                if status == "failed" and hasattr(scheduler, "fail"):
                    scheduler.fail(state.seq_id, reason, error)
                elif status == "cancelled" and hasattr(scheduler, "cancel"):
                    scheduler.cancel(state.seq_id, reason)
        self._enqueue_terminal(state)

    def _fail_request(self, state: _RequestState, error: str):
        if state.status in TERMINAL_STATUSES:
            return
        self._complete(state, "failed", "error", error)

    def _engine_fatal(self, exc: Exception):
        detail = f"engine-level failure: {type(exc).__name__}: {exc}"
        self._engine_error = detail
        self._closed = True
        for state in list(self._states.values()):
            if state.status not in TERMINAL_STATUSES:
                self._fail_request(state, detail)

    async def cancel(self, request_id: str, reason: str = "cancelled"):
        state = self._states.get(request_id)
        if state is None:
            return None
        if state.status in TERMINAL_STATUSES:
            return self.get_request_status(request_id)
        if reason not in {"cancelled", "timeout", "backpressure"}:
            reason = "cancelled"
        self._complete(state, "cancelled" if reason != "backpressure" else "failed", reason)
        return self.get_request_status(request_id)

    async def _expire_requests(self):
        now = monotonic()
        for request_id, state in list(self._states.items()):
            if state.status not in TERMINAL_STATUSES and now >= state.deadline:
                await self.cancel(request_id, reason="timeout")

    async def _run_loop(self):
        while not self._closed:
            try:
                await self._expire_requests()
                await self._drain_new_requests()
                try:
                    if self.engine.is_finished():
                        await asyncio.sleep(0.001)
                        continue
                    _, _, token_events = self.engine.step_with_events()
                    self._sync_sequence_statuses()
                except Exception as exc:
                    # A step failure cannot be attributed safely to one
                    # sequence. Stop the runner and fail all requests with an
                    # explicit engine-level diagnostic; do not retry in a loop.
                    self._engine_fatal(exc)
                    return
                for seq_id, token_id, finished in token_events:
                    state = self._streams.get(seq_id)
                    if state is None or state.status in {"cancelled", "failed"}:
                        continue
                    try:
                        self._record_first_token(state)
                        state.token_ids.append(token_id)
                        text = self.engine.tokenizer.decode(state.token_ids)
                        delta_text = text[len(state.text):]
                        state.text = text
                        if finished:
                            state.finish_reason = "stop"
                            state.status = "finished"
                            self._record_terminal_metrics(state)
                            terminal = RequestOutput(
                                request_id=state.request_id, text=state.text,
                                token_ids=list(state.token_ids), delta_text=delta_text,
                                delta_token_ids=[token_id], finished=True,
                                finish_reason="stop", status="finished",
                            )
                            self._streams.pop(seq_id, None)
                            self._seq_to_request.pop(seq_id, None)
                            self._clear_queue(state)
                            if not state.terminal_emitted:
                                state.queue.put_nowait(terminal)
                                state.terminal_emitted = True
                        else:
                            if state.queue.full():
                                await self.cancel(state.request_id, reason="backpressure")
                                continue
                            state.status = "decode"
                            state.queue.put_nowait(RequestOutput(
                                request_id=state.request_id, text=state.text,
                                token_ids=list(state.token_ids), delta_text=delta_text,
                                delta_token_ids=[token_id], finished=False,
                                finish_reason=None, status="decode",
                            ))
                    except Exception as exc:
                        self._fail_request(state, f"request-level failure: {type(exc).__name__}: {exc}")
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._engine_fatal(exc)
                return

    async def _drain_new_requests(self):
        while True:
            try:
                prompt, sampling_params, state = self._new_requests.get_nowait()
            except asyncio.QueueEmpty:
                return
            if state.status in TERMINAL_STATUSES:
                continue
            try:
                state.prompt_tokens = (
                    len(prompt)
                    if isinstance(prompt, list)
                    else len(self.engine.tokenizer.encode(prompt))
                )
                seq_id = self.engine.add_request(prompt, sampling_params)
                state.seq_id = seq_id
                self._streams[seq_id] = state
                self._seq_to_request[seq_id] = state.request_id
            except Exception as exc:
                self._complete(state, "failed", "error", f"{type(exc).__name__}: {exc}")
