"""Optional prebuilt W8A16 WMMA extension; this module never JIT-compiles."""
from __future__ import annotations

import torch

try:
    from . import _w8a16_tensorcore
    _IMPORT_ERROR = None
except Exception as exc:  # explicit fallback state, not a silent claim
    _w8a16_tensorcore = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def availability(x: torch.Tensor | None = None) -> tuple[bool, str]:
    if _w8a16_tensorcore is None:
        return False, f"CUDA WMMA extension is unavailable: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    device = x.device if x is not None and x.is_cuda else torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    if major * 10 + minor < 75:
        return False, f"SM{major}{minor} is below SM75"
    return True, "cuda_fp16_tensorcore_wmma"


def select_wmma_path(m: int, n: int, k: int) -> str:
    """Conservative policy measured on RTX 2080 Ti / SM75."""
    if m <= 16:
        return "cuda_fp16_tensorcore_wmma_decode_legacy"
    if m >= 256 and n >= 1024 and k >= 1024:
        return "cuda_fp16_tensorcore_wmma_prefill_cta"
    return "cuda_fp16_tensorcore_wmma_prefill_legacy"


def linear(x, weight_int8, weight_scale, bias=None):
    ok, reason = availability(x)
    if not ok:
        raise RuntimeError(reason)
    if x.dtype != torch.float16:
        raise RuntimeError("CUDA WMMA extension supports FP16 activation only")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((x2.shape[0], weight_int8.shape[0]), device=x.device, dtype=x.dtype)
    _w8a16_tensorcore.linear_cuda(x2, weight_int8.contiguous(), weight_scale.contiguous(), bias, out)
    return out.reshape(*x.shape[:-1], weight_int8.shape[0])


def linear_decode_wide_candidate(x, weight_int8, weight_scale, bias=None):
    """Benchmark-only 4-warp decode candidate; never used by production selector."""
    ok, reason = availability(x)
    if not ok:
        raise RuntimeError(reason)
    if x.dtype != torch.float16:
        raise RuntimeError("CUDA WMMA extension supports FP16 activation only")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    if x2.shape[0] > 16:
        raise ValueError("decode-wide candidate requires flattened M<=16")
    out = torch.empty((x2.shape[0], weight_int8.shape[0]), device=x.device, dtype=x.dtype)
    _w8a16_tensorcore.linear_cuda_decode_wide_candidate(
        x2, weight_int8.contiguous(), weight_scale.contiguous(), bias, out
    )
    linear_decode_wide_candidate.last_path = "cuda_fp16_tensorcore_wmma_decode_wide_candidate"
    return out.reshape(*x.shape[:-1], weight_int8.shape[0])
