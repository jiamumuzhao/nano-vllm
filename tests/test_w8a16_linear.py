import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.kernels.w8a16 import w8a16_linear
from nanovllm.kernels.w8a16_tensorcore import select_wmma_path, linear_decode_wide_candidate
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import reset_context, set_context
from transformers import Qwen3Config

from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    if not dist.is_initialized():
        path = tempfile.mktemp(prefix="nano-vllm-w8a16-")
        dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    yield


def _load_quantized(module, weight, bias=None):
    with torch.no_grad():
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
    module.quantize()
    return module


@pytest.mark.parametrize("shape", [(7, 11), (2, 5, 11)])
@pytest.mark.parametrize("bias", [False, True])
def test_column_w8a16_matches_fp16_and_supports_2d_3d(shape, bias):
    torch.manual_seed(0)
    out_features, in_features = 13, 11
    weight = torch.randn(out_features, in_features, dtype=torch.float16)
    b = torch.randn(out_features, dtype=torch.float16) if bias else None
    ref = F.linear(torch.randn(*shape, dtype=torch.float16), weight, b)
    x = torch.randn(*shape, dtype=torch.float16)
    # Reuse identical input for the reference.
    ref = F.linear(x, weight, b)
    module = _load_quantized(ColumnParallelLinear(in_features, out_features, bias=bias, quantization="w8a16"), weight, b)
    actual = module(x)
    torch.testing.assert_close(actual, ref, rtol=0.08, atol=0.08)
    assert module.weight_int8.dtype == torch.int8
    assert module.weight_scale.shape == (out_features,)
    assert not hasattr(module, "weight")
    assert module.weight_nbytes() < weight.numel() * weight.element_size()


def test_zero_output_channel_is_safe():
    weight = torch.randn(8, 9, dtype=torch.float16)
    weight[3].zero_()
    x = torch.randn(4, 9, dtype=torch.float16)
    module = _load_quantized(ColumnParallelLinear(9, 8, quantization="w8a16"), weight)
    assert module.weight_scale[3] != 0
    assert torch.count_nonzero(module.weight_int8[3]) == 0
    assert torch.count_nonzero(module(x)[:, 3]) == 0


@pytest.mark.parametrize("kind", ["merged", "qkv", "row"])
def test_parallel_linear_variants_and_gqa_shapes(kind):
    torch.manual_seed(1)
    if kind == "merged":
        module = MergedColumnParallelLinear(16, [12, 12], quantization="w8a16")
        weight = torch.randn(24, 16, dtype=torch.float16)
    elif kind == "qkv":
        # MHA/GQA packed local output: Q=4 heads, K/V=2 heads, head_dim=8.
        module = QKVParallelLinear(16, 8, 4, 2, quantization="w8a16")
        weight = torch.randn(64, 16, dtype=torch.float16)
    else:
        module = RowParallelLinear(16, 20, quantization="w8a16")
        weight = torch.randn(20, 16, dtype=torch.float16)
    x = torch.randn(3, 2, 16, dtype=torch.float16)
    _load_quantized(module, weight)
    actual = module(x)
    expected = F.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)
    assert actual.shape == (*x.shape[:-1], module.weight_int8.shape[0])
    assert module.weight_scale.shape[0] == module.weight_int8.shape[0]


def test_replicated_linear_remains_fp16():
    module = ReplicatedLinear(5, 7)
    weight = torch.randn(7, 5)
    with torch.no_grad():
        module.weight.copy_(weight)
    module.quantize()
    assert hasattr(module, "weight")
    assert not hasattr(module, "weight_int8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_qwen3_w8a16_logits_match_fp16():
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=128, attention_bias=True,
        tie_word_embeddings=False,
    )
    old_device = torch.get_default_device()
    old_dtype = torch.get_default_dtype()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)
    try:
        fp16_model = Qwen3ForCausalLM(cfg, {"quantization": "none"})
        w8a16_model = Qwen3ForCausalLM(cfg, {"quantization": "w8a16"})
        with torch.no_grad():
            for parameter in fp16_model.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        w8a16_model.load_state_dict(fp16_model.state_dict())
        for module in w8a16_model.modules():
            if hasattr(module, "quantize"):
                module.quantize()
        for model in (fp16_model, w8a16_model):
            for module in model.modules():
                if hasattr(module, "k_cache"):
                    module.k_cache = torch.empty(1, 128, 2, 64, device="cuda", dtype=torch.float16)
                    module.v_cache = torch.empty_like(module.k_cache)
        ids = torch.tensor([1, 2, 3, 4], device="cuda")
        positions = torch.arange(4, device="cuda")
        cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        slot = torch.arange(4, device="cuda", dtype=torch.int32)
        outputs = []
        for model in (fp16_model, w8a16_model):
            set_context(True, cu, cu, 4, 4, slot, None, None)
            with torch.inference_mode():
                outputs.append(model.compute_logits(model(ids, positions)))
            reset_context()
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0.35, atol=0.35)
    finally:
        reset_context()
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fp16_tile_kernel_non_aligned_gemm_2d_3d_bias_and_zero_channel():
    torch.manual_seed(17)
    out_features, in_features = 37, 1024
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
    weight[5].zero_()
    bias = torch.randn(out_features, device="cuda", dtype=torch.float16)
    module = ColumnParallelLinear(in_features, out_features, bias=True, quantization="w8a16").cuda().half()
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
    module.quantize()
    for x in (torch.randn(5, in_features, device="cuda", dtype=torch.float16),
              torch.randn(2, 3, in_features, device="cuda", dtype=torch.float16)):
        actual = module(x)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)
        from nanovllm.kernels.w8a16_tensorcore import availability
        supported, _ = availability(x)
        assert w8a16_linear.last_path == ("cuda_fp16_tensorcore_wmma_decode_legacy" if supported else "fp32_fma_fallback")
    assert torch.count_nonzero(module.weight_int8[5]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w8a16_wmma_uses_non_default_current_stream():
    torch.manual_seed(23)
    stream = torch.cuda.Stream()
    out_features, in_features, rows = 37, 1024, 5
    with torch.cuda.stream(stream):
        x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
        weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
        weight[4].zero_()
        bias = torch.randn(out_features, device="cuda", dtype=torch.float16)
        scale = weight.float().abs().amax(dim=1).clamp_min(1e-12).div(127).half()
        weight_int8 = torch.round(weight.float() / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
        actual = w8a16_linear(x, weight_int8, scale, bias)
        expected = F.linear(x.float(), weight_int8.float() * scale.float()[:, None], bias.float()).half()
        done = torch.cuda.Event()
        done.record(stream)
    # Synchronize only the non-default stream. A kernel incorrectly launched
    # on the default stream can therefore race this comparison.
    done.synchronize()
    assert w8a16_linear.last_path == "cuda_fp16_tensorcore_wmma_decode_legacy"
    torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 4, 16, 17, 64])
def test_w8a16_decode_and_prefill_cta_path_selection(m):
    torch.manual_seed(31 + m)
    n, k = 37, 1025
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    weight[3].zero_()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    scale = weight.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi = torch.round(weight.float() / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
    actual = w8a16_linear(x, wi, scale, bias)
    expected = F.linear(x.float(), wi.float() * scale.float()[:, None], bias.float()).half()
    expected_path = "cuda_fp16_tensorcore_wmma_decode_legacy" if m <= 16 else "cuda_fp16_tensorcore_wmma_prefill_legacy"
    assert w8a16_linear.last_path == expected_path
    torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)


def test_w8a16_selector_rules_are_explicit():
    assert select_wmma_path(1, 1024, 1024) == "cuda_fp16_tensorcore_wmma_decode_legacy"
    assert select_wmma_path(4, 4096, 1024) == "cuda_fp16_tensorcore_wmma_decode_legacy"
    assert select_wmma_path(16, 1024, 1024) == "cuda_fp16_tensorcore_wmma_decode_legacy"
    assert select_wmma_path(17, 1024, 1024) == "cuda_fp16_tensorcore_wmma_prefill_legacy"
    assert select_wmma_path(64, 1024, 1024) == "cuda_fp16_tensorcore_wmma_prefill_legacy"
    assert select_wmma_path(256, 1024, 1024) == "cuda_fp16_tensorcore_wmma_prefill_cta"
    assert select_wmma_path(256, 4096, 1024) == "cuda_fp16_tensorcore_wmma_prefill_cta"
    assert select_wmma_path(256, 1024, 512) == "cuda_fp16_tensorcore_wmma_prefill_legacy"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024), (256, 1024, 1024), (256, 4096, 1024)])
def test_w8a16_selected_prefill_paths_match_reference(m, n, k):
    torch.manual_seed(71 + m + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    weight[7].zero_()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    scale = weight.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi = torch.round(weight.float() / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
    actual = w8a16_linear(x, wi, scale, bias)
    expected = F.linear(x.float(), wi.float() * scale.float()[:, None], bias.float()).half()
    assert w8a16_linear.last_path == select_wmma_path(m, n, k)
    torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 4, 16])
@pytest.mark.parametrize("n,k", [(1024, 1024), (4096, 1024), (1024, 1031)])
def test_w8a16_decode_wide_candidate_matches_reference(m, n, k):
    torch.manual_seed(101 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    weight[11].zero_()
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    scale = weight.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi = torch.round(weight.float() / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
    actual = linear_decode_wide_candidate(x, wi, scale, bias)
    expected = F.linear(x.float(), wi.float() * scale.float()[:, None], bias.float()).half()
    torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w8a16_decode_wide_candidate_non_default_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        m, n, k = 4, 1024, 1031
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
        scale = weight.float().abs().amax(1).clamp_min(1e-12).div(127).half()
        wi = torch.round(weight.float() / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
        bias = torch.randn(n, device="cuda", dtype=torch.float16)
        actual = linear_decode_wide_candidate(x, wi, scale, bias)
        expected = F.linear(x.float(), wi.float() * scale.float()[:, None], bias.float()).half()
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.35, atol=0.35)


def test_cuda_kernel_has_fp16_dot_operands_static_regression():
    source = Path(__file__).resolve().parents[1].joinpath("nanovllm/kernels/w8a16.py").read_text()
    kernel_source = source.split(chr(10) + "def w8a16_linear", 1)[0]
    assert ".to(tl.float32)" not in kernel_source
    assert "out_dtype=tl.float32" not in kernel_source
    assert "dtype=tl.float32" in kernel_source
    assert "w_int8.to(tl.float16)" in kernel_source


def test_codegen_artifact_is_distinct_from_source_only_claim():
    import json
    artifacts = sorted(Path(__file__).resolve().parents[1].joinpath("docs/benchmarks").glob("w8a16_codegen_*.json"))
    assert artifacts, "run inspect_w8a16_codegen.py to create a PTX/SASS evidence artifact"
    data = json.loads(artifacts[-1].read_text())
    assert data["kernel_path"] in {"fp16_mma_verified", "fp32_fma_fallback", "unsupported"}
    assert "ptx_contains_fp16_mma" in data and "sass_contains_hmma" in data
    assert data["kernel_path"] != "fp16_mma_verified" or data["ptx_contains_fp16_mma"] or data["sass_contains_hmma"]
