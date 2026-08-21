import asyncio
from functools import wraps
from collections import deque
from types import SimpleNamespace

import pytest

import nanovllm.engine.async_engine as async_module
from nanovllm.engine.async_engine import AsyncEngine, QueueFullError
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams


class FakeScheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = deque()
        self._sequences = {}
        self.block_manager = SimpleNamespace(used_block_ids=set())

    def add(self, seq):
        seq.status = SequenceStatus.QUEUED
        self.waiting.append(seq)
        self._sequences[seq.seq_id] = seq

    def get_sequence(self, seq_id):
        return self._sequences.get(seq_id)

    def cancel(self, seq_id, reason="cancelled"):
        seq = self._sequences[seq_id]
        self.waiting = deque(x for x in self.waiting if x.seq_id != seq_id)
        self.running = deque(x for x in self.running if x.seq_id != seq_id)
        self.block_manager.used_block_ids.discard(seq_id)
        seq.status = SequenceStatus.CANCELLED
        seq.finish_reason = reason
        return True

    def fail(self, seq_id, reason="error", error=None):
        seq = self._sequences[seq_id]
        self.waiting = deque(x for x in self.waiting if x.seq_id != seq_id)
        self.running = deque(x for x in self.running if x.seq_id != seq_id)
        self.block_manager.used_block_ids.discard(seq_id)
        seq.status = SequenceStatus.FAILED
        seq.finish_reason = reason
        seq.error = error
        return True


class FakeEngine:
    def __init__(self, model, fail=False, blocked=False, **kwargs):
        self.scheduler = FakeScheduler()
        self.tokenizer = SimpleNamespace(decode=self._decode)
        self.fail = fail
        self.decode_fail_seq_id = None
        self.blocked = blocked
        self.exited = False
        self.next_id = 1

    def _decode(self, ids):
        if self.decode_fail_seq_id is not None and ids and ids[-1] == self.decode_fail_seq_id:
            raise ValueError("controlled tokenizer failure")
        return "".join(map(str, ids))

    def add_request(self, prompt, params):
        seq = SimpleNamespace(seq_id=self.next_id, status=SequenceStatus.QUEUED,
                              block_table=[], finish_reason=None, error=None,
                              max_tokens=params.max_tokens, produced=0)
        self.next_id += 1
        self.scheduler.add(seq)
        return seq.seq_id

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running or self.blocked

    def step_with_events(self):
        if self.fail:
            raise RuntimeError("fake runner failure")
        if self.scheduler.waiting:
            seq = self.scheduler.waiting.popleft()
            seq.status = SequenceStatus.DECODE
            self.scheduler.running.append(seq)
        seq = self.scheduler.running[0]
        seq.produced += 1
        finished = seq.produced >= seq.max_tokens
        if finished:
            seq.status = SequenceStatus.FINISHED
            self.scheduler.running.popleft()
        return [], -1, [(seq.seq_id, seq.seq_id, finished)]

    def exit(self):
        self.exited = True


async def _collect(generator):
    return [item async for item in generator]


def run_async(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return wrapped


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(async_module, "LLMEngine", FakeEngine)
    return async_module


@run_async
async def test_cancel_is_idempotent_and_releases_request(fake_engine):
    engine = AsyncEngine("fake", max_queue_size=4, stream_queue_size=2)
    task = asyncio.create_task(_collect(engine.generate([1], SamplingParams(max_tokens=4), request_id="r1")))
    await asyncio.sleep(0)
    first = await engine.cancel("r1")
    second = await engine.cancel("r1")
    assert first["status"] == "cancelled" and first["finish_reason"] == "cancelled"
    assert second == first
    output = await asyncio.wait_for(task, 1)
    assert output[-1].finish_reason == "cancelled"
    assert not engine.engine.scheduler.block_manager.used_block_ids
    await engine.shutdown()


@run_async
async def test_finished_status_and_stop_reason(fake_engine):
    engine = AsyncEngine("fake", stream_queue_size=2)
    output = await asyncio.wait_for(_collect(engine.generate([1], SamplingParams(max_tokens=2), request_id="r2")), 1)
    status = engine.get_request_status("r2")
    assert status["status"] == "finished"
    assert status["finish_reason"] == "stop"
    assert output[-1].status == "finished"
    assert engine.get_request_status("unknown") is None
    await engine.shutdown()


@run_async
async def test_timeout_and_runner_error_are_terminal(fake_engine):
    engine = AsyncEngine("fake", request_timeout_s=0.01)
    engine.engine.blocked = True
    output = await asyncio.wait_for(_collect(engine.generate([1], SamplingParams(max_tokens=2), request_id="timeout")), 1)
    assert output[-1].finish_reason == "timeout"
    assert engine.get_request_status("timeout")["status"] == "cancelled"
    await engine.shutdown()

    engine = AsyncEngine("fake")
    engine.engine.fail = True
    output = await asyncio.wait_for(_collect(engine.generate([1], SamplingParams(max_tokens=2), request_id="error")), 1)
    status = engine.get_request_status("error")
    assert output[-1].finish_reason == "error"
    assert status["status"] == "failed" and "fake runner failure" in status["error"]
    assert not engine.engine.scheduler.waiting
    await engine.shutdown()


@run_async
async def test_global_and_stream_backpressure(fake_engine):
    engine = AsyncEngine("fake", max_queue_size=1)
    first = engine.generate([1], SamplingParams(max_tokens=2), request_id="one")
    task = asyncio.create_task(_collect(first))
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        await _collect(engine.generate([1], SamplingParams(max_tokens=1), request_id="two"))
    await engine.cancel("one")
    await asyncio.wait_for(task, 1)
    await engine.shutdown()

    engine = AsyncEngine("fake", stream_queue_size=1)
    stream = engine.generate([1], SamplingParams(max_tokens=4), request_id="slow")
    # Prime the request, then deliberately stop consuming.  This exercises
    # the bounded per-request queue rather than racing the producer with a
    # fast consumer.
    first = await asyncio.wait_for(stream.__anext__(), 1)
    await asyncio.sleep(0.05)
    output = [first]
    try:
        while True:
            output.append(await asyncio.wait_for(stream.__anext__(), 1))
    except StopAsyncIteration:
        pass
    assert output[-1].finish_reason in {"backpressure", "cancelled"}
    assert engine.get_request_status("slow")["status"] == "failed"
    await engine.shutdown()


@run_async
async def test_shutdown_cleans_pending_and_active(fake_engine):
    engine = AsyncEngine("fake")
    task = asyncio.create_task(_collect(engine.generate([1], SamplingParams(max_tokens=20), request_id="active")))
    await asyncio.sleep(0)
    await engine.shutdown()
    await asyncio.wait_for(task, 1)
    assert engine.engine.exited
    assert engine.get_request_status("active")["status"] == "cancelled"
    await engine.shutdown()


@run_async
async def test_request_failure_isolated_from_other_request(fake_engine):
    engine = AsyncEngine("fake", stream_queue_size=4)
    engine.engine.decode_fail_seq_id = 1
    first = asyncio.create_task(_collect(engine.generate([1], SamplingParams(max_tokens=1), request_id="bad")))
    second = asyncio.create_task(_collect(engine.generate([2], SamplingParams(max_tokens=2), request_id="good")))
    bad, good = await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert bad[-1].finish_reason == "error"
    assert engine.get_request_status("bad")["status"] == "failed"
    assert good[-1].finish_reason == "stop"
    assert engine.get_request_status("good")["status"] == "finished"
    assert not engine.engine.scheduler.block_manager.used_block_ids
    assert not engine._streams and not engine._seq_to_request
    await engine.shutdown()


@run_async
async def test_engine_level_failure_is_fatal_and_does_not_retry(fake_engine):
    engine = AsyncEngine("fake")
    engine.engine.fail = True
    tasks = [asyncio.create_task(_collect(engine.generate([i], SamplingParams(max_tokens=3), request_id=f"r{i}")))
             for i in (1, 2)]
    outputs = await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert all(output[-1].finish_reason == "error" for output in outputs)
    assert all("engine-level failure" in engine.get_request_status(f"r{i}")["error"] for i in (1, 2))
    assert engine.is_unavailable()
    with pytest.raises(RuntimeError, match="engine-level failure"):
        await _collect(engine.generate([3], SamplingParams(max_tokens=1), request_id="new"))
    await engine.shutdown()


@run_async
async def test_terminal_protocol_has_no_sentinel_tasks_and_cancel_cleans_maps(fake_engine):
    engine = AsyncEngine("fake", stream_queue_size=1)
    output = await asyncio.wait_for(_collect(engine.generate([1], SamplingParams(max_tokens=1), request_id="terminal")), 1)
    assert output[-1].finished and output[-1].finish_reason == "stop"
    assert not engine._streams and not engine._seq_to_request
    assert not any(
        task is not asyncio.current_task() and not task.done() and getattr(task.get_coro(), "__name__", "") == "put"
        for task in asyncio.all_tasks()
    )

    blocked = AsyncEngine("fake", stream_queue_size=1)
    blocked.engine.blocked = True
    pending = asyncio.create_task(_collect(blocked.generate([1], SamplingParams(max_tokens=4), request_id="queued")))
    await asyncio.sleep(0)
    await blocked.cancel("queued")
    await asyncio.wait_for(pending, 1)
    assert not blocked._streams and not blocked._seq_to_request
    assert not blocked.engine.scheduler.block_manager.used_block_ids
    await blocked.shutdown()
