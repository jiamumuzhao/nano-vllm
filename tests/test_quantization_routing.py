from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner


def _config(quantization):
    fake_hf = SimpleNamespace(max_position_embeddings=8192)
    with patch("nanovllm.config.os.path.isdir", return_value=True), \
         patch("nanovllm.config.AutoConfig.from_pretrained", return_value=fake_hf):
        return Config("/test/model", quantization=quantization)


def test_none_and_w8a16_are_the_only_production_configurations():
    assert _config("none").quantization == "none"
    assert _config("w8a16").quantization == "w8a16"


@pytest.mark.parametrize("quantization", ["w8a8", "w8a8_cublaslt", "w8a8_experimental", "w8a8_int8_mma"])
def test_w8a8_variants_are_rejected_at_config_validation(quantization):
    with pytest.raises(ValueError, match="benchmark-only") as exc:
        _config(quantization)
    message = str(exc.value)
    assert "none" in message
    assert "w8a16" in message


def test_model_runner_quantization_metadata_keeps_w8a16_explicit():
    none = SimpleNamespace(quantization="none", enforce_eager=False)
    w8a16 = SimpleNamespace(quantization="w8a16", enforce_eager=False)
    assert ModelRunner.quantization_metadata(none)["route"] == "fp16"
    assert ModelRunner.quantization_metadata(none)["role"] == "default FP16 performance baseline"
    assert ModelRunner.quantization_metadata(w8a16)["route"] == "w8a16"
    assert "memory" in ModelRunner.quantization_metadata(w8a16)["role"]
    assert ModelRunner.effective_enforce_eager(none) is False
    assert ModelRunner.effective_enforce_eager(w8a16) is True
    with pytest.raises(ValueError, match="benchmark-only"):
        ModelRunner.quantization_metadata(SimpleNamespace(quantization="w8a8"))
