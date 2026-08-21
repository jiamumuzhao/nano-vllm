"""Bounded TP startup protocol tests that do not require a model or CUDA."""

import multiprocessing as mp

import pytest

from nanovllm.engine.model_runner import wait_for_worker_ready_events


def _exit_without_ready():
    return


def test_unready_worker_is_reported_and_reaped():
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_exit_without_ready)
    process.start()
    try:
        with pytest.raises(RuntimeError, match="TP worker readiness timeout") as error:
            wait_for_worker_ready_events(
                [ready], [process], "tcp://127.0.0.1:29999", timeout=5.0,
            )
        message = str(error.value)
        assert "unready_ranks=[1]" in message
        assert str(process.pid) in message
        assert "exitcode" in message
        assert "tcp://127.0.0.1:29999" in message
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    assert not process.is_alive()
