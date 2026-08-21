import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

SPLIT_KV_BUCKETS = (1, 2, 4, 8, 16, 32)




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
def _paged_prefill_kernel_serial(
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
    q_tile_id = tl.program_id(2)

    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    seq_q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    q_start = seq_q_start + q_tile_id * BLOCK_Q
    if q_start >= seq_q_end:
        return
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

@triton.jit
def _decode_paged_split_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, partial_max_ptr, partial_sum_ptr, partial_acc_ptr,
    block_tables_ptr, context_lens_ptr, scale,
    stride_q_b, stride_q_h, stride_q_d,
    stride_kc_blk, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_blk, stride_vc_t, stride_vc_h, stride_vc_d,
    stride_pm_b, stride_pm_h, stride_pm_p, stride_ps_b, stride_ps_h, stride_ps_p,
    stride_pa_b, stride_pa_h, stride_pa_p, stride_pa_d, stride_bt_b,
    BLOCK_KV: tl.constexpr, BLOCK_SIZE: tl.constexpr, PARTITION_SIZE: tl.constexpr,
    D: tl.constexpr, N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    partition_id = tl.program_id(2)
    seq_len = tl.load(context_lens_ptr + batch_id)
    partition_start = partition_id * PARTITION_SIZE
    partition_end = tl.minimum(seq_len, partition_start + PARTITION_SIZE)
    if N_HEADS == N_KV_HEADS:
        kv_head_id = head_id
    else:
        kv_head_id = head_id // (N_HEADS // N_KV_HEADS)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + batch_id * stride_q_b + head_id * stride_q_h + offs_d * stride_q_d)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    # Local online-softmax state; reduction rescales it with the global maximum.
    for kv_start in range(partition_start, partition_end, BLOCK_KV):
        logical_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = logical_offsets < partition_end
        block_idx = logical_offsets // BLOCK_SIZE
        block_offset = logical_offsets - block_idx * BLOCK_SIZE
        block_id = tl.load(block_tables_ptr + batch_id * stride_bt_b + block_idx, mask=kv_mask, other=0)
        k = tl.load(k_cache_ptr + block_id[:, None] * stride_kc_blk + block_offset[:, None] * stride_kc_t + kv_head_id * stride_kc_h + offs_d[None, :] * stride_kc_d, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_cache_ptr + block_id[:, None] * stride_vc_blk + block_offset[:, None] * stride_vc_t + kv_head_id * stride_vc_h + offs_d[None, :] * stride_vc_d, mask=kv_mask[:, None], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(kv_mask, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(partial_max_ptr + batch_id * stride_pm_b + head_id * stride_pm_h + partition_id * stride_pm_p, m_i)
    tl.store(partial_sum_ptr + batch_id * stride_ps_b + head_id * stride_ps_h + partition_id * stride_ps_p, l_i)
    tl.store(partial_acc_ptr + batch_id * stride_pa_b + head_id * stride_pa_h + partition_id * stride_pa_p + offs_d * stride_pa_d, acc)

@triton.jit
def _reduce_split_kv_kernel(
    partial_max_ptr, partial_sum_ptr, partial_acc_ptr, o_ptr, num_partitions,
    stride_pm_b, stride_pm_h, stride_pm_p, stride_ps_b, stride_ps_h, stride_ps_p,
    stride_pa_b, stride_pa_h, stride_pa_p, stride_pa_d, stride_o_b, stride_o_h, stride_o_d,
    D: tl.constexpr, MAX_PARTITIONS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offs_d = tl.arange(0, D)
    global_max = float("-inf")
    for partition_id in range(0, MAX_PARTITIONS):
        valid = partition_id < num_partitions
        m = tl.load(partial_max_ptr + batch_id * stride_pm_b + head_id * stride_pm_h + partition_id * stride_pm_p, mask=valid, other=float("-inf"))
        global_max = tl.maximum(global_max, m)
    total_sum = 0.0
    total_acc = tl.zeros([D], dtype=tl.float32)
    for partition_id in range(0, MAX_PARTITIONS):
        valid = partition_id < num_partitions
        m = tl.load(partial_max_ptr + batch_id * stride_pm_b + head_id * stride_pm_h + partition_id * stride_pm_p, mask=valid, other=float("-inf"))
        s = tl.load(partial_sum_ptr + batch_id * stride_ps_b + head_id * stride_ps_h + partition_id * stride_ps_p, mask=valid, other=0.0)
        a = tl.load(partial_acc_ptr + batch_id * stride_pa_b + head_id * stride_pa_h + partition_id * stride_pa_p + offs_d * stride_pa_d, mask=valid, other=0.0)
        weight = tl.where(m == float("-inf"), 0.0, tl.exp(m - global_max))
        total_sum += s * weight
        total_acc += a * weight
    o = total_acc / total_sum
    tl.store(o_ptr + batch_id * stride_o_b + head_id * stride_o_h + offs_d * stride_o_d, o.to(o_ptr.dtype.element_ty))


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
        split_kv_config=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.enable_gqa = num_heads != num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        cfg = split_kv_config or {}
        self.split_kv_enabled = cfg.get("split_kv_enabled", True)
        self.split_kv_threshold = cfg.get("split_kv_threshold", 1024)
        self.split_kv_partition_size = cfg.get("split_kv_partition_size", 1024)
        self.split_kv_max_partitions = cfg.get("split_kv_max_partitions", 16)
        mode = cfg.get("paged_prefill_q_tile_mode", "auto")
        legacy_parallel = cfg.get("paged_prefill_q_tile_parallel")
        if mode == "auto" and legacy_parallel is not None:
            # Backward compatibility: an explicitly supplied legacy bool wins
            # over auto, preserving its old forced serial/parallel behavior.
            mode = "parallel" if legacy_parallel else "serial"
        if mode not in ("serial", "parallel", "auto"):
            raise ValueError("paged_prefill_q_tile_mode must be serial, parallel, or auto")
        self.paged_prefill_q_tile_mode = mode
        self.paged_prefill_auto_min_batch_size = cfg.get("paged_prefill_auto_min_batch_size", 8)
        self.paged_prefill_auto_min_q_tiles = cfg.get("paged_prefill_auto_min_q_tiles", 128)
        self.paged_prefill_auto_parallel_enabled = cfg.get("paged_prefill_auto_parallel_enabled", False)
        # Benchmark-only overrides; None preserves the production heuristic.
        self.paged_prefill_block_q = cfg.get("paged_prefill_block_q")
        self.paged_prefill_block_kv = cfg.get("paged_prefill_block_kv")
        self.paged_prefill_num_warps = cfg.get("paged_prefill_num_warps")
        self._split_workspace = None

    def initialize_split_workspace(self, max_batch_size: int, device: torch.device,
                                    max_partitions: int | None = None):
        max_partitions = max_partitions or self.split_kv_max_partitions
        shape = (max_batch_size, self.num_heads, max_partitions)
        acc_shape = shape + (self.head_dim,)
        self._split_workspace = (
            torch.empty(shape, device=device, dtype=torch.float32),
            torch.empty(shape, device=device, dtype=torch.float32),
            torch.empty(acc_shape, device=device, dtype=torch.float32),
        )

    def _split_partition_count(self, context_len: int) -> int:
        requested = max(1, (context_len + self.split_kv_partition_size - 1) // self.split_kv_partition_size)
        for bucket in SPLIT_KV_BUCKETS:
            if requested <= bucket:
                if bucket > self.split_kv_max_partitions:
                    break
                return bucket
        raise ValueError(
            f"Split-KV context length {context_len} requires {requested} partitions, "
            f"exceeding split_kv_max_partitions={self.split_kv_max_partitions}"
        )

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

    def _select_paged_prefill_path(self, batch_size: int, max_seqlen_q: int,
                                   block_q: int) -> bool:
        """Return whether to use Q-tile parallel paged prefill.

        The latest formal 2026-07-20 Qwen3-0.6B eager matrix on an RTX 2080 Ti
        showed no clear end-to-end parallel gain across its 18 workloads, so
        auto safely falls back to serial. The batch/tile thresholds remain
        configurable for a future revalidated matrix. Explicit serial/parallel
        modes always override auto.
        """
        if self.paged_prefill_q_tile_mode == "serial":
            return False
        if self.paged_prefill_q_tile_mode == "parallel":
            return True
        q_tile_count = (max_seqlen_q + block_q - 1) // block_q
        return (self.paged_prefill_auto_parallel_enabled
                and batch_size >= self.paged_prefill_auto_min_batch_size
                and q_tile_count >= self.paged_prefill_auto_min_q_tiles)

    def _prefill_paged(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                       k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
        block_size = k_cache.size(1)
        assert self.head_dim in (64, 128, 256)
        assert context.block_tables is not None
        o = torch.empty_like(q)
        batch_size = context.cu_seqlens_q.numel() - 1
        max_seqlen_q = context.max_seqlen_q or q.size(0)
        if max_seqlen_q <= 1:
            block_q, block_kv, num_warps = 1, 64, 4
        elif max_seqlen_q <= 16:
            block_q, block_kv, num_warps = 16, 64, 4
        else:
            block_q, block_kv, num_warps = 32, 64, 4
        if self.paged_prefill_block_q is not None:
            block_q = self.paged_prefill_block_q
        if self.paged_prefill_block_kv is not None:
            block_kv = self.paged_prefill_block_kv
        if self.paged_prefill_num_warps is not None:
            num_warps = self.paged_prefill_num_warps
        max_q_tiles = (max_seqlen_q + block_q - 1) // block_q
        use_parallel = self._select_paged_prefill_path(batch_size, max_seqlen_q, block_q)
        if use_parallel:
            grid = (batch_size, self.num_heads, max_q_tiles)
            kernel = _paged_prefill_kernel
        else:
            grid = (batch_size, self.num_heads)
            kernel = _paged_prefill_kernel_serial
        kernel[grid](
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
        graph_capture = torch.cuda.is_current_stream_capturing()
        # CUDA Graph capture intentionally keeps Split-KV disabled: the graph path
        # must not inspect a device tensor with .item(), change its grid, or allocate.
        max_context_len = int(context.context_lens.max().item()) if not graph_capture else 0
        use_split = (self.split_kv_enabled and not graph_capture
                     and max_context_len >= self.split_kv_threshold)
        if use_split:
            num_partitions = self._split_partition_count(max_context_len)
            if self._split_workspace is None:
                raise RuntimeError("Split-KV workspace was not initialized before decode")
            partial_max, partial_sum, partial_acc = self._split_workspace
            partial_max = partial_max[:bs]
            partial_sum = partial_sum[:bs]
            partial_acc = partial_acc[:bs]
            _decode_paged_split_kernel[(bs, self.num_heads, num_partitions)](
                q, k_cache, v_cache, partial_max, partial_sum, partial_acc,
                context.block_tables, context.context_lens, self.scale,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
                partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
                partial_sum.stride(0), partial_sum.stride(1), partial_sum.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                context.block_tables.stride(0),
                BLOCK_KV=64, BLOCK_SIZE=block_size, PARTITION_SIZE=self.split_kv_partition_size,
                D=self.head_dim, N_HEADS=self.num_heads, N_KV_HEADS=self.num_kv_heads,
            )
            _reduce_split_kv_kernel[(bs, self.num_heads)](
                partial_max, partial_sum, partial_acc, o, num_partitions,
                partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
                partial_sum.stride(0), partial_sum.stride(1), partial_sum.stride(2),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                o.stride(0), o.stride(1), o.stride(2),
                D=self.head_dim, MAX_PARTITIONS=self.split_kv_max_partitions,
            )
            return o

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
