import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_logits_against_transformers.py"
spec = importlib.util.spec_from_file_location("verify_logits_against_transformers", SCRIPT)
verify = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verify
spec.loader.exec_module(verify)


def test_valid_token_mask_excludes_mixed_batch_padding():
    mask = verify.valid_token_mask([17, 32, 127])
    assert tuple(mask.shape) == (3, 127)
    assert mask.sum(dim=1).tolist() == [17, 32, 127]
    assert not mask[0, 17]
    assert not mask[1, 32]


def test_metrics_and_thresholds_cover_full_vocabulary_logits():
    expected = torch.zeros(2, 3, 5)
    actual = expected.clone()
    actual[0, 1, 0] = 0.01
    metrics = verify.compute_metrics(actual, expected)
    assert metrics["elements"] == 30
    assert metrics["max_abs_error"] == pytest.approx(0.01)
    assert metrics["top1_token_agreement"] == 1.0
    assert verify.passes_thresholds(metrics, {"max_abs_error": 0.05, "mean_abs_error": 0.005, "top1_token_agreement": 1.0})

    actual[1, 0, 4] = 0.2
    failed = verify.compute_metrics(actual, expected)
    assert not verify.passes_thresholds(failed, {"max_abs_error": 0.05, "mean_abs_error": 0.005, "top1_token_agreement": 1.0})


def test_masked_metrics_do_not_count_padding_rows():
    expected = torch.zeros(2, 4, 3)
    actual = expected.clone()
    actual[0, 3, 0] = 10.0
    metrics = verify.compute_metrics(actual, expected, verify.valid_token_mask([3, 4], 4))
    assert metrics["logit_rows"] == 7
    assert metrics["max_abs_error"] == 0.0


def test_packed_to_padded_and_layerwise_record_are_serializable():
    packed = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    padded = verify.packed_to_padded(packed, [1, 2])
    assert tuple(padded.shape) == (2, 2, 2)
    assert padded[0, 0].tolist() == [0.0, 1.0]
    assert padded[1, 1].tolist() == [4.0, 5.0]
    record = verify.layerwise_record(0, "prefill", "embedding", padded, padded, verify.valid_token_mask([1, 2], 2))
    assert record["record_type"] == "layerwise"
    assert record["metrics"]["elements"] == 6


def test_layer2_boundary_records_have_complete_metrics_and_location():
    names = [
        "layer_2.attention_residual_input",
        "layer_2.post_attention_layernorm",
        "layer_2.gate_up_proj",
        "layer_2.silu_activation",
        "layer_2.down_proj",
        "layer_2.decoder_layer_output",
    ]
    records = [verify.layerwise_record(0, "prefill", name, torch.ones(2, 3), torch.zeros(2, 3)) for name in names]
    assert {r["layer"] for r in records} == set(names)
    for record in records:
        assert record["metrics"]["elements"] == 6
        assert {"max_abs_error", "mean_abs_error", "rmse", "top1_token_agreement"} <= record["metrics"].keys()
        assert {"row", "feature", "actual", "expected"} <= record["max_error_location"].keys()
    first = verify.first_layerwise_exceedance(records)
    assert first["layer"] == names[0]


def test_v3_full_boundary_order_and_metadata_reject_shape_mismatch():
    assert verify.V2_BOUNDARY_ORDER.index("layer_input") < verify.V2_BOUNDARY_ORDER.index("logits")
    actual = torch.ones(1, 2, 4)
    expected = torch.zeros_like(actual)
    record = verify.layerwise_v3_record(0, "prefill", "layer_0", "layer_input", actual, expected)
    assert record["schema"] == "logits_alignment_v3"
    assert record["shape"] == {"nano": [1, 2, 4], "transformers": [1, 2, 4]}
    assert record["layout"].startswith("[batch,tokens")
    with pytest.raises(ValueError, match="shape mismatch"):
        verify.layerwise_v3_record(0, "prefill", "layer_0", "q_projection", actual, torch.zeros(1, 2, 5))


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_qk_rmsnorm_is_per_head_and_matches_hf_order(head_dim):
    from nanovllm.layers.layernorm import RMSNorm

    torch.manual_seed(head_dim)
    x = torch.randn(2, 3, head_dim, dtype=torch.float16)
    # Exercise a non-contiguous view without changing the logical last dim.
    x = x.transpose(0, 1).contiguous().transpose(0, 1)
    weight = torch.linspace(0.5, 1.5, head_dim, dtype=torch.float16)
    norm = RMSNorm(head_dim, eps=1e-6).eval()
    norm.weight.data = weight.clone()
    actual = norm(x)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.float16) * weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Different heads must be normalized independently; a cross-head reduction
    # would change this result when one head is rescaled.
    x2 = x.clone()
    x2[:, 0] *= 100
    out2 = norm(x2)
    ref2 = (x2.float() * torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16) * weight
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_v3_qk_norm_metadata_has_input_output_and_per_head_summary():
    actual = torch.zeros(1, 2, 3, 4)
    expected = torch.zeros_like(actual)
    record = verify.layerwise_v3_record(
        0, "prefill", "layer_1", "k_norm_output", actual, expected,
        verify.valid_token_mask([2], 2), (actual, expected),
    )
    assert record["normalization_dim"] == -1
    assert record["input_shape"]["nano"] == [1, 2, 3, 4]
    assert record["output_shape"]["transformers"] == [1, 2, 3, 4]
    assert len(record["per_head_error_summary"]) == 3
    assert record["normalization_semantics"].startswith("RMSNorm over head_dim")


def test_debug_split_gates_is_explicit_and_production_default_is_merged():
    parser = verify.build_parser()
    default = parser.parse_args(["--model", "/tmp/model"])
    split = parser.parse_args(["--model", "/tmp/model", "--validation-split-gates"])
    assert default.debug_layerwise is False or default.validation_split_gates is False
    assert default.validation_split_gates is False
    assert split.validation_split_gates is True


def test_fp16_rmsnorm_swiglu_and_residual_match_reference_order():
    from nanovllm.layers.activation import SiluAndMul
    from nanovllm.layers.layernorm import RMSNorm

    x = torch.tensor([[0.25, -1.5, 2.0, 0.75]], dtype=torch.float16)
    residual = torch.tensor([[1.0, 0.5, -0.25, 2.0]], dtype=torch.float16)
    norm = RMSNorm(4, eps=1e-6).eval()
    norm.weight.data = norm.weight.data.to(torch.float16)
    actual_norm, actual_residual = norm(x.clone(), residual.clone())
    combined = x + residual
    variance = combined.float().pow(2).mean(dim=-1, keepdim=True)
    expected_norm = (combined.float() * torch.rsqrt(variance + 1e-6)).to(torch.float16) * norm.weight
    assert torch.equal(actual_residual, combined)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)

    gate_up = torch.tensor([[0.5, -1.0, 2.0, 0.25, 1.5, -0.5, 0.75, 2.0]], dtype=torch.float16)
    actual_swiglu = SiluAndMul()(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    expected_swiglu = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual_swiglu, expected_swiglu, rtol=0, atol=0)


def test_token_progression_is_shared_and_tp_is_rejected():
    assert [verify.deterministic_token(0, 1, step, 1000) for step in range(8)] == [
        verify.deterministic_token(0, 1, step, 1000) for step in range(8)
    ]
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        verify.validate_tensor_parallel_size(2)


def test_json_and_markdown_serialization_is_auditable(tmp_path):
    metrics = {
        "elements": 6,
        "logit_rows": 1,
        "vocabulary_size": 6,
        "max_abs_error": 0.01,
        "mean_abs_error": 0.001,
        "rmse": 0.002,
        "top1_token_agreement": 1.0,
    }
    record = {"case": 0, "lengths": [17, 32, 127], "phase": "prefill", "metrics": metrics, "passed": True}
    env = {"model": "local", "gpu": "test", "torch": "test", "cuda_runtime": None}
    thresholds = {"max_abs_error": 0.05, "mean_abs_error": 0.005, "top1_token_agreement": 1.0}
    text = verify.markdown([record], {"passed": True}, env, thresholds)
    path = tmp_path / "result.jsonl"
    verify.atomic_write(path, json.dumps({"record_type": "case", **record}) + "\n")
    assert path.read_text().startswith('{"record_type"')
    assert "TP=1 correctness" in text
    assert "[17, 32, 127]" in text


@pytest.mark.gpu
@pytest.mark.integration
def test_gpu_qwen3_logits_alignment_when_available():
    model = Path("/root/huggingface/Qwen3-0.6B")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; GPU logits alignment is not a CPU test")
    if not model.is_dir():
        pytest.skip("local Qwen3-0.6B model is unavailable; no download is attempted")
    args = verify.build_parser().parse_args([
        "--model", str(model),
        "--tensor-parallel-size", "1",
        "--lengths", "1,16,17,32,128",
        "--mixed-lengths", "17,32,127",
        "--decode-steps", "4",
        "--output-dir", str(model.parent / "nano-vllm-logits-test-artifacts"),
    ])
    assert verify.run(args) == 0
