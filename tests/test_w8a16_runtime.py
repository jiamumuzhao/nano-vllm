from types import SimpleNamespace

from nanovllm.engine.model_runner import ModelRunner


def _config(quantization, enforce_eager=False):
    return SimpleNamespace(quantization=quantization, enforce_eager=enforce_eager)


def test_w8a16_forces_eager_and_does_not_capture():
    config = _config("w8a16", enforce_eager=False)
    assert ModelRunner.effective_enforce_eager(config) is True
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.enforce_eager = ModelRunner.effective_enforce_eager(config)
    runner.cuda_graph_disabled_reason = "w8a16 eager-only"
    assert runner.enforce_eager is True
    assert runner.should_capture_cudagraph() is False
    assert not hasattr(runner, "graphs")
    assert not hasattr(runner, "graph_pool")


def test_fp16_graph_selection_is_unchanged():
    config = _config("none", enforce_eager=False)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.enforce_eager = ModelRunner.effective_enforce_eager(config)
    assert runner.enforce_eager is False
    assert runner.should_capture_cudagraph() is True
