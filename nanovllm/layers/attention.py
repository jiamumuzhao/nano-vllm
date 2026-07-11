import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)# key.stride(0)==num_heads*head_dim


# ---------------------------------------------------------------------------
# Varlen prefill: one (seq, head) program iterates over the seq's Q tiles
# and scans the full KV range with online safe softmax.
# ---------------------------------------------------------------------------

@triton.jit
def _varlen_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    scale,
    stride_q_t, stride_q_h, stride_q_d,# token,num_heads,head_dim
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Sequence boundaries in the 1D layout
    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    seq_q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    seq_k_start = tl.load(cu_seqlens_k_ptr + seq_id)
    seq_k_end = tl.load(cu_seqlens_k_ptr + seq_id + 1)

    seq_q_len = seq_q_end - seq_q_start
    seq_k_len = seq_k_end - seq_k_start
    prefix_len = seq_k_len - seq_q_len

    # GQA: map query head to KV head
    if N_HEADS == N_KV_HEADS:
        kv_head_id = head_id
    else:
        kv_head_id = head_id // (N_HEADS // N_KV_HEADS)

    offs_d = tl.arange(0, D)

    # Tile over query positions within this sequence
    for q_start in range(seq_q_start, seq_q_end, BLOCK_Q):
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < seq_q_end

        # Load Q tile  (BLOCK_Q, D)
        q = tl.load(
            q_ptr + offs_q[:, None] * stride_q_t + head_id * stride_q_h + offs_d[None, :] * stride_q_d,
            mask=q_mask[:, None],
            other=0.0,
        )

        # Online safe softmax state
        m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

        # Tile over KV positions
        for kv_start in range(seq_k_start, seq_k_end, BLOCK_KV):
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = offs_kv < seq_k_end

            # Load K, V tile  (BLOCK_KV, D)
            k = tl.load(
                k_ptr + offs_kv[:, None] * stride_k_t + kv_head_id * stride_k_h + offs_d[None, :] * stride_k_d,
                mask=kv_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_ptr + offs_kv[:, None] * stride_v_t + kv_head_id * stride_v_h + offs_d[None, :] * stride_v_d,
                mask=kv_mask[:, None],
                other=0.0,
            )

            # Scores  (BLOCK_Q, BLOCK_KV)
            scores = tl.dot(q, tl.trans(k))
            scores *= scale

            if CAUSAL:
                q_local = offs_q - seq_q_start          # (BLOCK_Q,)
                k_local = offs_kv - seq_k_start          # (BLOCK_KV,)
                causal_mask = k_local[None, :] <= q_local[:, None] + prefix_len
                valid_mask = q_mask[:, None] & kv_mask[None, :]
                scores = tl.where(valid_mask & causal_mask, scores, float("-inf"))

            # Online safe softmax step
            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new

        # Normalise and store
        o = acc / l_i[:, None]
        tl.store(
            o_ptr + offs_q[:, None] * stride_o_t + head_id * stride_o_h + offs_d[None, :] * stride_o_d,
            o.to(o_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )


# ---------------------------------------------------------------------------
# Paged prefill/decode kernels read KV from page tables and only touch valid
# logical tokens.
# ---------------------------------------------------------------------------

@triton.jit
def _paged_prefill_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr, block_tables_ptr,
    scale,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kc_blk, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_blk, stride_vc_t, stride_vc_h, stride_vc_d,
    stride_bt_b,
    stride_o_t, stride_o_h, stride_o_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    seq_q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    seq_k_start = tl.load(cu_seqlens_k_ptr + seq_id)
    seq_k_end = tl.load(cu_seqlens_k_ptr + seq_id + 1)

    seq_q_len = seq_q_end - seq_q_start
    seq_k_len = seq_k_end - seq_k_start
    prefix_len = seq_k_len - seq_q_len

    if N_HEADS == N_KV_HEADS:
        kv_head_id = head_id
    else:
        kv_head_id = head_id // (N_HEADS // N_KV_HEADS)

    offs_d = tl.arange(0, D)

    for q_start in range(seq_q_start, seq_q_end, BLOCK_Q):
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < seq_q_end
        q = tl.load(
            q_ptr + offs_q[:, None] * stride_q_t + head_id * stride_q_h + offs_d[None, :] * stride_q_d,
            mask=q_mask[:, None],
            other=0.0,
        )

        m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

        for kv_start in range(0, seq_k_len, BLOCK_KV):
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = offs_kv < seq_k_len
            block_idx = offs_kv // BLOCK_SIZE
            block_offset = offs_kv - block_idx * BLOCK_SIZE
            block_id = tl.load(
                block_tables_ptr + seq_id * stride_bt_b + block_idx,
                mask=kv_mask,
                other=0,
            )

            k = tl.load(
                k_cache_ptr + block_id[:, None] * stride_kc_blk + block_offset[:, None] * stride_kc_t + kv_head_id * stride_kc_h + offs_d[None, :] * stride_kc_d,
                mask=kv_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_cache_ptr + block_id[:, None] * stride_vc_blk + block_offset[:, None] * stride_vc_t + kv_head_id * stride_vc_h + offs_d[None, :] * stride_vc_d,
                mask=kv_mask[:, None],
                other=0.0,
            )

            scores = tl.dot(q, tl.trans(k)) * scale
            q_local = offs_q - seq_q_start
            causal_mask = offs_kv[None, :] <= q_local[:, None] + prefix_len
            valid_mask = q_mask[:, None] & kv_mask[None, :]
            scores = tl.where(valid_mask & causal_mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new

        o = acc / l_i[:, None]
        tl.store(
            o_ptr + offs_q[:, None] * stride_o_t + head_id * stride_o_h + offs_d[None, :] * stride_o_d,
            o.to(o_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )


@triton.jit
def _decode_paged_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    block_tables_ptr, context_lens_ptr,
    scale,
    stride_q_b, stride_q_h, stride_q_d,
    stride_kc_blk, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_blk, stride_vc_t, stride_vc_h, stride_vc_d,
    stride_bt_b,
    stride_o_b, stride_o_h, stride_o_d,
    BLOCK_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_len = tl.load(context_lens_ptr + batch_id)
    if seq_len <= 0:
        return

    # GQA
    if N_HEADS == N_KV_HEADS:
        kv_head_id = head_id
    else:
        kv_head_id = head_id // (N_HEADS // N_KV_HEADS)

    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    offs_d = tl.arange(0, D)

    # Load single query vector  (D,)
    q = tl.load(q_ptr + batch_id * stride_q_b + head_id * stride_q_h + offs_d * stride_q_d)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    '''
    block->tile
    '''

    # Iterate over pages
    for blk_idx in range(0, num_blocks):
        blk_id = tl.load(block_tables_ptr + batch_id * stride_bt_b + blk_idx)
        blk_start = blk_idx * BLOCK_SIZE
        tokens_in_blk = tl.minimum(BLOCK_SIZE, seq_len - blk_start)

        # Tile within the block
        for t_start in range(0, tokens_in_blk, BLOCK_KV):
            kv_len = tl.minimum(BLOCK_KV, tokens_in_blk - t_start)
            offs_t = t_start + tl.arange(0, BLOCK_KV)
            kv_mask = offs_t < tokens_in_blk

            # Load K tile
            k = tl.load(
                k_cache_ptr + blk_id * stride_kc_blk + offs_t[:, None] * stride_kc_t + kv_head_id * stride_kc_h + offs_d[None, :] * stride_kc_d,
                mask=kv_mask[:, None],
                other=0.0,
            )
            # Load V tile
            v = tl.load(
                v_cache_ptr + blk_id * stride_vc_blk + offs_t[:, None] * stride_vc_t + kv_head_id * stride_vc_h + offs_d[None, :] * stride_vc_d,
                mask=kv_mask[:, None],
                other=0.0,
            )

            # Score  (BLOCK_KV,)
            scores = tl.sum(q[None, :] * k, axis=1)
            scores *= scale
            scores = tl.where(kv_mask, scores, float("-inf"))

            # Online safe softmax (scalar m_i / l_i since 1 query)
            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

    o = acc / l_i
    tl.store(
        o_ptr + batch_id * stride_o_b + head_id * stride_o_h + offs_d * stride_o_d,
        o.to(o_ptr.dtype.element_ty),
    )


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.enable_gqa = num_heads != num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # cu_seqlens_k[-1] > cu_seqlens_q[-1] means cached prefix exists
            if context.cu_seqlens_k[-1] > context.cu_seqlens_q[-1]:
                o = self._prefill_paged(q, k, v, k_cache, v_cache, context)
            else:
                o = self._prefill_varlen(q, k, v, context)
        else:
            o = self._decode_paged(q, k_cache, v_cache, context)
        return o

    # ------------------------------------------------------------------
    # Varlen prefill (dense – no prefix)
    # ------------------------------------------------------------------

    def _prefill_varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        assert self.head_dim in (64, 128, 256), "head_dim must be a power of 2"
        batch_size = context.cu_seqlens_q.numel() - 1
        o = torch.empty_like(q)

        grid = (batch_size, self.num_heads)
        _varlen_prefill_kernel[grid](
            q, k, v, o,
            context.cu_seqlens_q, context.cu_seqlens_k,
            self.scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_Q=32, BLOCK_KV=64,
            D=self.head_dim,
            N_HEADS=self.num_heads,
            N_KV_HEADS=self.num_kv_heads,
            CAUSAL=True,
        )
        return o

    # ------------------------------------------------------------------
    # Paged prefill (prefix cache) -- read full K/V directly from KV cache.
    # ------------------------------------------------------------------

    def _prefill_paged(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                       k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
        block_size = k_cache.size(1)
        assert self.head_dim in (64, 128, 256)
        assert context.block_tables is not None
        o = torch.empty_like(q)
        batch_size = context.cu_seqlens_q.numel() - 1
        grid = (batch_size, self.num_heads)
        max_seqlen_q = context.max_seqlen_q or q.size(0)
        if max_seqlen_q <= 1:
            block_q, block_kv, num_warps = 1, 64, 4
        elif max_seqlen_q <= 16:
            block_q, block_kv, num_warps = 16, 64, 4
        else:
            block_q, block_kv, num_warps = 32, 64, 4
        _paged_prefill_kernel[grid](
            q, k_cache, v_cache, o,
            context.cu_seqlens_q, context.cu_seqlens_k, context.block_tables,
            self.scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            context.block_tables.stride(0),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_Q=block_q,
            BLOCK_KV=block_kv,
            BLOCK_SIZE=block_size,
            D=self.head_dim,
            N_HEADS=self.num_heads,
            N_KV_HEADS=self.num_kv_heads,
            num_warps=num_warps,
        )
        return o

    # ------------------------------------------------------------------
    # Paged decode – read from page table, never pad
    # ------------------------------------------------------------------

    def _decode_paged(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
        bs = q.size(0)
        block_size = k_cache.size(1)
        assert self.head_dim in (64, 128, 256)

        o = torch.empty_like(q)

        grid = (bs, self.num_heads)
        _decode_paged_kernel[grid](
            q, k_cache, v_cache, o,
            context.block_tables, context.context_lens,
            self.scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            context.block_tables.stride(0),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_KV=64,
            BLOCK_SIZE=block_size,
            D=self.head_dim,
            N_HEADS=self.num_heads,
            N_KV_HEADS=self.num_kv_heads,
        )
        return o
