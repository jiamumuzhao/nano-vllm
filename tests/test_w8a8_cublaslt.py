import pytest
import torch

from nanovllm.kernels.w8a8_cublaslt import PATH, W8A8Workspace, linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_case(m, n, k, bias=True):
    torch.manual_seed(730000 + m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    wf = torch.randn(n, k, device="cuda", dtype=torch.float16)
    sw = wf.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi = torch.round(wf.float() / sw.float()[:, None]).clamp(-127, 127).to(torch.int8)
    b = torch.randn(n, device="cuda", dtype=torch.float16) if bias else None
    sx = x.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    xq = torch.round(x.float() / sx.float()[:, None]).clamp(-127, 127)
    ref = (xq @ wi.float().t()) * sx.float()[:, None] * sw.float()[None, :]
    if b is not None:
        ref = ref + b.float()
    return x, wi, sw, b, ref.half()


@pytest.mark.parametrize("m", [1, 4, 16, 32, 64, 256])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("bias", [False, True])
def test_cublaslt_shapes_and_path(m, n, bias):
    k = 1024
    x, wi, sw, b, ref = _make_case(m, n, k, bias)
    workspace = W8A8Workspace(m, n, k, x.device)
    workspace.prepare(wi, shapes=[(m, n, k)])
    ptrs = workspace.data_ptrs()
    y = linear(x, wi, sw, b, workspace=workspace)
    y2 = linear(x, wi, sw, b, workspace=workspace)
    assert linear.last_path == PATH
    torch.testing.assert_close(y, ref, rtol=.08, atol=.60)
    torch.testing.assert_close(y2, ref, rtol=.08, atol=.60)
    assert workspace.data_ptrs() == ptrs


def test_cublaslt_non_aligned_or_explicit_unsupported():
    m, n, k = 17, 1025, 1031
    x, wi, sw, b, ref = _make_case(m, n, k, True)
    workspace = W8A8Workspace(m, n, k, x.device)
    try:
        workspace.prepare(wi, shapes=[(m, n, k)])
        y = linear(x, wi, sw, b, workspace=workspace)
        assert linear.last_path == PATH
        torch.testing.assert_close(y, ref, rtol=.08, atol=.60)
    except (RuntimeError, ValueError) as exc:
        assert "cuBLASLt" in str(exc) or "cublasLt" in str(exc) or "shape" in str(exc)


def test_cublaslt_zero_row_zero_channel_and_3d():
    m, n, k = 4, 1024, 1024
    x, wi, sw, b, _ = _make_case(m, n, k, True)
    x.zero_()
    wi[7].zero_()
    workspace = W8A8Workspace(m, n, k, x.device)
    workspace.prepare(wi, shapes=[(m, n, k)])
    y = linear(x.view(1, 4, k), wi, sw, b, workspace=workspace)
    assert linear.last_path == PATH
    expected = b.view(1, 1, n).expand(1, 4, n)
    torch.testing.assert_close(y, expected, rtol=.0, atol=.01)
    assert torch.count_nonzero(y[..., 7] - b[7]) == 0


def test_cublaslt_non_default_stream_and_persistent_ptrs():
    m, n, k = 32, 4096, 1024
    x, wi, sw, b, ref = _make_case(m, n, k, True)
    workspace = W8A8Workspace(m, n, k, x.device)
    workspace.prepare(wi, shapes=[(m, n, k)])
    before = workspace.data_ptrs()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = linear(x, wi, sw, b, workspace=workspace)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    assert linear.last_path == PATH
    torch.testing.assert_close(y, ref, rtol=.08, atol=.60)
    assert workspace.data_ptrs() == before


def test_cublaslt_workspace_validation():
    x, wi, sw, b, _ = _make_case(4, 1024, 1024, True)
    with pytest.raises(ValueError, match="requires CUDA device"):
        W8A8Workspace(4, 1024, 1024, torch.device("cpu"))
    workspace = W8A8Workspace(4, 1024, 1024, x.device)
    workspace.prepare(wi, shapes=[(4, 1024, 1024)])
    with pytest.raises(ValueError, match="dtype"):
        linear(x.float(), wi, sw, b, workspace=workspace)
    with pytest.raises(ValueError, match="capacity"):
        linear(torch.randn(8, 1024, device="cuda", dtype=torch.float16), wi, sw, b, workspace=workspace)


def test_cublaslt_rejects_different_weight_until_reprepare():
    x, w1, sw1, b, _ = _make_case(16, 1024, 1024, True)
    _, w2, sw2, _, ref2 = _make_case(16, 1024, 1024, True)
    w2[0, 0] = 1 if int(w2[0, 0].item()) != 1 else 2
    sx2 = x.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    xq2 = torch.round(x.float() / sx2.float()[:, None]).clamp(-127, 127)
    ref2 = ((xq2 @ w2.float().t()) * sx2.float()[:, None] * sw2.float()[None, :] + b.float()).half()
    assert w1.data_ptr() != w2.data_ptr()
    workspace = W8A8Workspace(16, 1024, 1024, x.device)
    workspace.prepare(w1, shapes=[(16, 1024, 1024)])
    with pytest.raises(ValueError, match="packed weight|prepare"):
        linear(x, w2, sw2, b, workspace=workspace)
    workspace.prepare(w2, shapes=[(16, 1024, 1024)])
    y = linear(x, w2, sw2, b, workspace=workspace)
    assert linear.last_path == PATH
    torch.testing.assert_close(y, ref2, rtol=.08, atol=.60)


def test_cublaslt_rejects_in_place_weight_mutation_until_reprepare():
    x, w1, sw1, b, _ = _make_case(16, 1024, 1024, True)
    workspace = W8A8Workspace(16, 1024, 1024, x.device)
    workspace.prepare(w1, shapes=[(16, 1024, 1024)])
    ptr = w1.data_ptr()
    old = int(w1[0, 0].item())
    w1[0, 0] = 1 if old != 1 else 2
    assert w1.data_ptr() == ptr
    with pytest.raises(ValueError, match="packed weight|prepare|version"):
        linear(x, w1, sw1, b, workspace=workspace)
    workspace.prepare(w1, shapes=[(16, 1024, 1024)])
    sx = x.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    xq = torch.round(x.float() / sx.float()[:, None]).clamp(-127, 127)
    ref = ((xq @ w1.float().t()) * sx.float()[:, None] * sw1.float()[None, :] + b.float()).half()
    y = linear(x, w1, sw1, b, workspace=workspace)
    torch.testing.assert_close(y, ref, rtol=.08, atol=.60)


def test_cublaslt_autotune_records_candidates_and_layout_decision():
    m, n, k = 16, 1024, 1024
    x, wi, sw, b, ref = _make_case(m, n, k, True)
    workspace = W8A8Workspace(m, n, k, x.device)
    workspace.prepare(wi, shapes=[(m, n, k)])
    shape = (m, n, k)
    selected = workspace.selected_metadata[shape]
    candidates = workspace.candidates[shape]
    assert candidates
    assert selected["layout"] == "row_major"
    assert selected["algorithm"]
    assert selected["workspace_bytes"] >= 0
    assert selected["p50_ms"] >= 0
    assert all("status" in candidate for candidate in candidates)
    col32 = workspace.layout_results[shape]["col32"]
    assert col32["attempted"] is True
    assert col32["stage"] in {"heuristic", "matmul", "transform"}
    assert col32.get("status_code") is not None
    assert col32.get("cuBLASLt_status") or col32.get("status")
    assert col32.get("reason") or col32.get("success") is True

    baseline = W8A8Workspace(m, n, k, x.device)
    baseline.prepare(wi, shapes=[shape], autotune=False)
    selected_ptrs = workspace.data_ptrs()
    baseline_ptrs = baseline.data_ptrs()
    y_selected = linear(x, wi, sw, b, workspace=workspace)
    y_baseline = linear(x, wi, sw, b, workspace=baseline)
    torch.testing.assert_close(y_selected, ref, rtol=.08, atol=.60)
    torch.testing.assert_close(y_baseline, ref, rtol=.08, atol=.60)
    torch.testing.assert_close(y_selected, y_baseline, rtol=.02, atol=.05)
    assert workspace.data_ptrs() == selected_ptrs
    assert baseline.data_ptrs() == baseline_ptrs
    assert baseline.selected_metadata[shape]["layout"] == "row_major"
