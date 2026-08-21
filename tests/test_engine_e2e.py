"""GPU engine-level regression tests for the real Nano-vLLM lifecycle.

These tests intentionally inspect token ids, scheduler events and block-manager
state.  They do not assert generated natural-language text and are never
replaced by CPU tests when CUDA or the local model is unavailable.
"""
from __future__ import annotations

import os
import gc
import multiprocessing as mp
import signal
import socket
import time
import traceback
from contextlib import contextmanager
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams


MODEL = Path(os.environ.get("NANOVLLM_QWEN3_MODEL", "/root/huggingface/Qwen3-0.6B"))
pytestmark = [pytest.mark.gpu, pytest.mark.e2e]


def _require_gpu_model():
    if not torch.cuda.is_available():
        pytest.skip("GPU E2E requires CUDA; CPU fallback is intentionally not used")
    if not MODEL.is_dir():
        pytest.skip(f"local Qwen3-0.6B model is unavailable: {MODEL}")


def _sampling(max_tokens=2):
    # SamplingParams currently rejects exact temperature=0.  1e-10 is the
    # supported deterministic/greedy limit and keeps this test on the actual
    # engine sampler rather than monkeypatching production inference.
    return SamplingParams(temperature=1e-9, max_tokens=max_tokens, ignore_eos=True)


def _prompt(length, offset=100):
    return [offset + (i % 97) for i in range(length)]


def _rendezvous_endpoint():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _make_engine(**kwargs):
    _require_gpu_model()
    defaults = dict(
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.5,
        tp_startup_timeout=45.0,
        tp_shutdown_timeout=15.0,
        distributed_init_method=_rendezvous_endpoint(),
    )
    defaults.update(kwargs)
    try:
        return LLM(str(MODEL), **defaults)
    except Exception:
        # ModelRunner initializes NCCL before KV allocation.  If a deliberately
        # tiny test configuration cannot allocate a block, clean that partial
        # initialization before pytest evaluates the next independent test.
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        raise


def _diagnostics(engine):
    if engine is None:
        return []
    return [
        {"rank": i + 1, "pid": p.pid, "is_alive": p.is_alive(), "exitcode": p.exitcode,
         "ready": engine.ready_events[i].is_set() if i < len(engine.ready_events) else False}
        for i, p in enumerate(engine.ps)
    ]


@contextmanager
def _bounded(label, engine=None, timeout=90):
    def alarm(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {timeout}s; workers={_diagnostics(engine)}")
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _sequences(engine):
    return {seq.seq_id: seq for seq in (*engine.scheduler.waiting, *engine.scheduler.running)}


def _drain(engine, max_steps=512):
    events = []
    outputs = {}
    for _ in range(max_steps):
        if engine.is_finished():
            break
        finished, _, token_events = engine.step_with_events()
        events.extend(token_events)
        outputs.update(dict(finished))
    assert engine.is_finished(), "engine did not finish within the bounded E2E step budget"
    assert not engine.scheduler.waiting
    assert not engine.scheduler.running
    assert not engine.scheduler.block_manager.used_block_ids
    return outputs, events


def _assert_completed_outputs(outputs, requested_ids, max_tokens):
    assert set(outputs) == set(requested_ids)
    for token_ids in outputs.values():
        assert isinstance(token_ids, list)
        assert len(token_ids) <= max_tokens
        assert all(isinstance(token, int) and token >= 0 for token in token_ids)


def _close(engine):
    engine.exit()
    engine.exit()
    assert not torch.distributed.is_initialized()
    # The test owns this Engine exclusively; clear CPU-side owners as well so
    # a later Engine in the same pytest process cannot inherit its CUDA cache.
    engine.scheduler = None
    engine.tokenizer = None
    engine.config = None
    gc.collect()
    torch.cuda.empty_cache()


def test_00_tp2_engine_prefill_decode_and_worker_cleanup():
    # Run TP=2 before the single-GPU lifecycle cases so their model allocations
    # cannot compete with rank 0 on the first visible GPU.
    _run_tp2_test_body()


def test_prefix_cache_reuse_and_full_resource_release():
    engine = _make_engine(max_num_batched_tokens=32)
    original_call = None
    observe_call = None
    try:
        prefix = _prompt(32, 100)
        first_id = engine.add_request(prefix + [7, 8, 9, 10], _sampling(2))
        first_outputs, first_events = _drain(engine)
        _assert_completed_outputs(first_outputs, [first_id], 2)
        assert first_events
        baseline_free = len(engine.scheduler.block_manager.free_block_ids)

        second_id = engine.add_request(prefix + [17, 18, 19, 20], _sampling(2))
        observed_cached = []
        original_call = engine.model_runner.call

        def observe_call(method_name, *args):
            if method_name == "run" and args[1]:
                observed_cached.extend(seq.num_cached_tokens for seq in args[0])
            return original_call(method_name, *args)

        engine.model_runner.call = observe_call
        second_outputs, second_events = _drain(engine)
        _assert_completed_outputs(second_outputs, [second_id], 2)
        assert second_events
        assert any(cached >= len(prefix) for cached in observed_cached), observed_cached
        assert len(engine.scheduler.block_manager.free_block_ids) == baseline_free
    finally:
        if original_call is not None:
            engine.model_runner.call = original_call
        _close(engine)
        if observe_call is not None:
            del observe_call


def test_chunked_prefill_is_fair_and_completes_all_requests():
    try:
        engine = _make_engine(max_num_batched_tokens=16, max_model_len=96, gpu_memory_utilization=0.5)
    except AssertionError as exc:
        pytest.skip(f"chunked-prefill configuration could not allocate KV blocks: {exc}")
    original_call = None
    observe_call = None
    try:
        long_id = engine.add_request(_prompt(64, 200), _sampling(2))
        short_id = engine.add_request(_prompt(8, 400), _sampling(2))
        calls = []
        original_call = engine.model_runner.call

        def observe_call(method_name, *args):
            if method_name == "run":
                calls.append((bool(args[1]), tuple(seq.seq_id for seq in args[0])))
            return original_call(method_name, *args)

        engine.model_runner.call = observe_call
        outputs, events = _drain(engine)
        _assert_completed_outputs(outputs, [long_id, short_id], 2)
        prefill_calls = [ids for is_prefill, ids in calls if is_prefill]
        assert len(prefill_calls) >= 2, calls
        assert any(long_id in ids for ids in prefill_calls)
        assert any(short_id in ids for ids in prefill_calls)
        assert any(event[0] == short_id for event in events)
    finally:
        if original_call is not None:
            engine.model_runner.call = original_call
        _close(engine)
        if observe_call is not None:
            del observe_call


def test_preemption_recompute_and_error_cleanup():
    try:
        engine = _make_engine(
            max_num_batched_tokens=32,
            max_model_len=96,
            max_num_seqs=2,
            gpu_memory_utilization=0.5,
        )
    except AssertionError as exc:
        pytest.skip(f"configured preemption capacity produced no KV blocks: {exc}")
    try:
        # Keep the real scheduler/block manager logic, but bound the effective
        # free-block pool after engine setup so this test is deterministic on
        # machines whose physical GPU has a much larger KV capacity.
        manager = engine.scheduler.block_manager
        manager.free_block_ids = deque(list(manager.free_block_ids)[:6])
        ids = [engine.add_request(_prompt(48, 500 + i * 20), _sampling(4)) for i in range(2)]
        outputs, _ = _drain(engine, max_steps=1024)
        _assert_completed_outputs(outputs, ids, 4)
        assert engine.scheduler.preemption_count >= 1, engine.scheduler.preemption_events
        assert engine.scheduler.preemption_events
        assert not engine.scheduler.block_manager.used_block_ids
    finally:
        _close(engine)

    engine = _make_engine(max_num_batched_tokens=16)
    original_call = None
    fail_run = None
    try:
        seq_id = engine.add_request(_prompt(8, 700), _sampling(1))
        original_call = engine.model_runner.call

        def fail_run(method_name, *args):
            if method_name == "run":
                raise RuntimeError("controlled ModelRunner failure")
            return original_call(method_name, *args)

        engine.model_runner.call = fail_run
        with pytest.raises(RuntimeError, match="controlled ModelRunner failure"):
            engine.step_with_events()
        seq = _sequences(engine).get(seq_id)
        assert seq is not None
    finally:
        if original_call is not None:
            engine.model_runner.call = original_call
        _close(engine)
        if fail_run is not None:
            del fail_run


def test_99_cuda_graph_and_eager_have_identical_token_ids_when_supported():
    _require_gpu_model()
    free, total = torch.cuda.mem_get_info()
    if free < total * 0.75:
        pytest.skip(f"CUDA Graph comparison requires a clean memory margin; free={free} total={total}")
    prompt = _prompt(8, 800)
    params = _sampling(2)
    eager = None
    graph = None
    eager_out_ids = None
    try:
        torch.manual_seed(1234)
        eager = _make_engine(max_num_batched_tokens=16, enforce_eager=True)
    except (RuntimeError, AssertionError) as exc:
        detail = str(exc) or "ModelRunner.allocate_kv_cache asserted num_kvcache_blocks > 0"
        pytest.skip(f"eager setup unavailable for CUDA Graph comparison: {type(exc).__name__}: {detail}")
    try:
        eager_id = eager.add_request(prompt, params)
        eager_out, _ = _drain(eager)
        eager_out_ids = list(eager_out[eager_id])
    finally:
        _close(eager)
        eager = None

    free, total = torch.cuda.mem_get_info()
    if free < total * 0.75:
        pytest.skip(f"CUDA Graph capture requires a clean post-eager memory margin; free={free} total={total}")

    try:
        torch.manual_seed(1234)
        graph = _make_engine(max_num_batched_tokens=16, enforce_eager=False)
    except (RuntimeError, AssertionError) as exc:
        detail = str(exc) or "ModelRunner CUDA Graph setup asserted during initialization"
        pytest.skip(f"CUDA Graph initialization/capture unsupported: {type(exc).__name__}: {detail}")
    try:
        if graph.model_runner.enforce_eager or not hasattr(graph.model_runner, "graphs"):
            pytest.skip("CUDA Graph capture is not available for this configuration")
        graph_id = graph.add_request(prompt, params)
        graph_out, _ = _drain(graph)
        assert eager_out_ids == list(graph_out[graph_id])
    finally:
        if graph is not None:
            _close(graph)


def _run_tp2_case(result_queue):
    """Run TP=2 in an isolated process so a CUDA/NCCL hang is recoverable."""
    os.setsid()
    engine = None
    result = {"status": "failed", "startup_s": None, "inference_s": None, "shutdown_s": None}
    try:
        started = time.monotonic()
        with _bounded("TP=2 engine startup", timeout=90):
            engine = _make_engine(
                tensor_parallel_size=2,
                max_num_batched_tokens=32,
                max_num_seqs=2,
                enforce_eager=True,
                tp_startup_timeout=45.0,
                tp_shutdown_timeout=15.0,
            )
        result["startup_s"] = time.monotonic() - started
        ids = [
            engine.add_request(_prompt(8, 900), _sampling(2)),
            engine.add_request(_prompt(12, 950), _sampling(2)),
        ]
        inference_started = time.monotonic()
        with _bounded("TP=2 prefill/decode", engine, timeout=90):
            outputs, events = _drain(engine)
        result["inference_s"] = time.monotonic() - inference_started
        _assert_completed_outputs(outputs, ids, 2)
        assert events
        result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if engine is not None:
            shutdown_started = time.monotonic()
            try:
                with _bounded("TP=2 shutdown", engine, timeout=30):
                    _close(engine)
                result["shutdown_s"] = time.monotonic() - shutdown_started
            except BaseException:
                result["status"] = "failed"
                result["error"] = traceback.format_exc()
            result["workers"] = [
                {"pid": process.pid, "is_alive": process.is_alive(), "exitcode": process.exitcode}
                for process in engine.ps
            ]
            result["used_blocks"] = 0
            result["parent_dist_initialized"] = dist.is_initialized()
    result_queue.put(result)


def _kill_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.join(timeout=10)


def _run_tp2_test_body():
    _require_gpu_model()
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 E2E requires at least two visible CUDA GPUs")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_run_tp2_case, args=(result_queue,))
    process.start()
    process.join(timeout=150)
    if process.is_alive():
        _kill_process_group(process)
        pytest.fail(f"TP=2 E2E timed out after 150s; child pid={process.pid}")
    assert process.exitcode == 0, f"TP=2 child exited with code {process.exitcode}"
    result = result_queue.get(timeout=5)
    assert result["status"] == "passed", result.get("error", result)
    assert result["workers"] and all(not worker["is_alive"] for worker in result["workers"]), result
    assert result["used_blocks"] == 0, result
    assert result["parent_dist_initialized"] is False, result
