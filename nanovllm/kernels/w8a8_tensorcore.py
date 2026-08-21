"""Benchmark-only W8A8 INT8 Tensor Core path; never used by production routing."""
from __future__ import annotations
import torch

try:
    from . import _w8a8_tensorcore
    _IMPORT_ERROR = None
except Exception as exc:
    _w8a8_tensorcore = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def availability(x=None):
    if _w8a8_tensorcore is None:
        return False, f"W8A8 extension unavailable: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    dev = x.device if x is not None and x.is_cuda else torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(dev)
    if major * 10 + minor < 75:
        return False, f"SM{major}{minor} is below SM75"
    return True, "w8a8_experimental_int8_mma"


def quantize_activation(x):
    ok, reason = availability(x)
    if not ok: raise RuntimeError(reason)
    if x.dtype != torch.float16: raise RuntimeError("W8A8 experimental path supports FP16 activation only")
    x2=x.reshape(-1,x.shape[-1]).contiguous()
    xq=torch.empty_like(x2,dtype=torch.int8)
    xs=torch.empty((x2.shape[0],),device=x.device,dtype=torch.float16)
    _w8a8_tensorcore.quantize_cuda(x2,xq,xs)
    return xq,xs


def _select_implementation(m, n, k, implementation):
    """Return an explicit kernel choice; auto is deliberately conservative.

    No exact shape has passed the W8A8 future-integration gate, so auto keeps
    the production-like benchmark baseline on the legacy kernel. CTA choices
    remain explicit benchmark/test candidates.
    """
    if implementation not in {"auto", "legacy", "cta16x64", "cta32x64"}:
        raise ValueError(f"unknown W8A8 implementation: {implementation}")
    if implementation == "auto" or implementation == "legacy":
        return "legacy"
    if implementation == "cta16x64":
        return "cta16x64"
    return "cta32x64"


def linear(x, weight_int8, weight_scale, bias=None, implementation="auto"):
    xq,xs=quantize_activation(x)
    out=torch.empty((*xq.shape[:-1],weight_int8.shape[0]),device=x.device,dtype=torch.float16)
    m,n,k=xq.shape[0],weight_int8.shape[0],xq.shape[1]
    selected = _select_implementation(m, n, k, implementation)
    if selected == "cta16x64":
        _w8a8_tensorcore.gemm_cuda_cta16(xq,xs,weight_int8.contiguous(),weight_scale.contiguous(),bias,out)
        linear.last_path="w8a8_experimental_int8_mma_cta16x64"
    elif selected == "cta32x64":
        _w8a8_tensorcore.gemm_cuda_cta32(xq,xs,weight_int8.contiguous(),weight_scale.contiguous(),bias,out)
        linear.last_path="w8a8_experimental_int8_mma_cta32x64"
    else:
        _w8a8_tensorcore.gemm_cuda(xq,xs,weight_int8.contiguous(),weight_scale.contiguous(),bias,out)
        linear.last_path="w8a8_experimental_int8_mma"
    return out.reshape(*x.shape[:-1],weight_int8.shape[0])


linear.last_path="uninitialized"
