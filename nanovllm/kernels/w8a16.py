import torch
import triton
import triton.language as tl

try:
    from nanovllm.kernels.w8a16_tensorcore import availability as _wmma_availability
    from nanovllm.kernels.w8a16_tensorcore import linear as _wmma_linear
    from nanovllm.kernels.w8a16_tensorcore import select_wmma_path as _select_wmma_path
except Exception as exc:
    _wmma_availability = lambda x=None: (False, f"WMMA extension import failed: {exc}")
    _wmma_linear = None
    _select_wmma_path = lambda m, n, k: "fp32_fma_fallback"

_WMMA_DIAGNOSTIC_EMITTED = False



@triton.jit
def _w8a16_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    m_size, n_size, k_size,
    sxm, sxk, swn, swk, sscale, sbias,
    som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k_size, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_fp16 = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < k_size),
            other=0.0,
        ).to(tl.float16)
        w_int8 = tl.load(
            w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk,
            mask=(offs_n[:, None] < n_size) & (offs_k[None, :] < k_size),
            other=0,
        )
        scale_fp16 = tl.load(
            scale_ptr + offs_n * sscale, mask=offs_n < n_size, other=0.0
        ).to(tl.float16)
        # Dequantize only this resident tile. The dot operands are explicitly
        # FP16 while only the accumulator is FP32.
        w_fp16 = (w_int8.to(tl.float16) * scale_fp16[:, None]).to(tl.float16)
        acc += tl.dot(x_fp16, tl.trans(w_fp16))
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n * sbias, mask=offs_n < n_size, other=0.0)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        acc,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def w8a16_linear(x: torch.Tensor, weight_int8: torch.Tensor,
                 weight_scale: torch.Tensor, bias: torch.Tensor | None):
    original_shape = x.shape
    if x.ndim not in (2, 3):
        raise ValueError(f"W8A16 Linear expects 2D or 3D activation, got {x.ndim}D")
    global _WMMA_DIAGNOSTIC_EMITTED
    if not (x.is_cuda and weight_int8.is_cuda and x.dtype in (torch.float16, torch.bfloat16)):
        # Correctness fallback: FP32 matmul with int8 weights and per-channel scale,
        # without constructing a full dequantized weight tensor.
        y = torch.matmul(x.float(), weight_int8.t().float())
        y = y * weight_scale.float()
        if bias is not None:
            y = y + bias.float()
        w8a16_linear.last_path = "fp32_matmul_fallback"
        return y.to(x.dtype)
    if x.dtype == torch.float16 and _wmma_linear is not None:
        supported, reason = _wmma_availability(x)
        if supported:
            try:
                flat_m = x.reshape(-1, x.shape[-1]).shape[0]
                w8a16_linear.last_path = _select_wmma_path(
                    flat_m, weight_int8.shape[0], x.shape[-1]
                )
                return _wmma_linear(x, weight_int8, weight_scale, bias)
            except Exception as exc:
                reason = f"WMMA extension execution failed: {type(exc).__name__}: {exc}"
        if not _WMMA_DIAGNOSTIC_EMITTED:
            print(f"W8A16 WMMA unavailable; using fp32_fma_fallback: {reason}")
            _WMMA_DIAGNOSTIC_EMITTED = True
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    m_size, k_size = x2.shape
    n_size = weight_int8.shape[0]
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    if k_size == 0 or n_size == 0:
        return out.reshape(*original_shape[:-1], n_size)
    block_m, block_n, block_k = 32, 32, 64
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    w8a16_linear.last_path = "fp32_fma_fallback"
    _w8a16_kernel[grid](
        x2, weight_int8, weight_scale, bias, out,
        m_size, n_size, k_size,
        x2.stride(0), x2.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        weight_scale.stride(0), 0 if bias is None else bias.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return out.reshape(*original_shape[:-1], n_size)
