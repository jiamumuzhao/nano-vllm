import torch
from torch import nn
import torch.nn.functional as F
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
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
            if context.block_tables is not None:    # prefix cache
                o = self._prefill_paged(q, k, v, k_cache, v_cache, context)
            else:
                o = self._prefill_dense(q, k, v, context)
        else:    # decode
            o = self._decode(q, k_cache, v_cache, context)
        return o

    def _prefill_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        cu_q = context.cu_seqlens_q.tolist()
        outputs = []
        for i in range(len(cu_q) - 1):
            s, e = cu_q[i], cu_q[i + 1]
            q_i = q[s:e].transpose(0, 1).unsqueeze(0)     # (1, H,   T, D)
            k_i = k[s:e].transpose(0, 1).unsqueeze(0)     # (1, Hkv, T, D)
            v_i = v[s:e].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                is_causal=True,
                scale=self.scale,
                enable_gqa=self.enable_gqa,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))    # (T, H, D)
        return torch.cat(outputs, dim=0)
    '''
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=self.model_dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
                '''

    def _prefill_paged(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
        block_size = k_cache.size(1)
        num_kv_heads = k_cache.size(2)
        head_dim = k_cache.size(3)
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        block_tables = context.block_tables
        outputs = []
        for i in range(len(cu_q) - 1):
            qs, qe = cu_q[i], cu_q[i + 1]
            seqlen_q = qe - qs
            seqlen_k = cu_k[i + 1] - cu_k[i]
            prefix_len = seqlen_k - seqlen_q
            if prefix_len > 0:
                num_prefix_blocks = (prefix_len + block_size - 1) // block_size
                bt = block_tables[i, :num_prefix_blocks].clamp(min=0)
                k_prefix = k_cache[bt].reshape(-1, num_kv_heads, head_dim)[:prefix_len]
                v_prefix = v_cache[bt].reshape(-1, num_kv_heads, head_dim)[:prefix_len]
                k_i = torch.cat([k_prefix, k_new[qs:qe]], dim=0)
                v_i = torch.cat([v_prefix, v_new[qs:qe]], dim=0)
            else:
                k_i = k_new[qs:qe]
                v_i = v_new[qs:qe]
            q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
            k_i = k_i.transpose(0, 1).unsqueeze(0)
            v_i = v_i.transpose(0, 1).unsqueeze(0)
            q_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
            k_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
            attn_mask = k_idx <= (q_idx + prefix_len)
            out = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                attn_mask=attn_mask,
                scale=self.scale,
                enable_gqa=self.enable_gqa,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def _decode(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context) -> torch.Tensor:
        bs = q.size(0)
        block_size = k_cache.size(1)
        num_kv_heads = k_cache.size(2)
        head_dim = k_cache.size(3)
        block_tables = context.block_tables
        context_lens = context.context_lens
        max_blocks = block_tables.size(1)
        max_len = max_blocks * block_size

        bt_safe = block_tables.clamp(min=0)
        k = k_cache[bt_safe].reshape(bs, max_len, num_kv_heads, head_dim)
        v = v_cache[bt_safe].reshape(bs, max_len, num_kv_heads, head_dim)

        pos = torch.arange(max_len, device=q.device, dtype=context_lens.dtype)
        valid = pos.unsqueeze(0) < context_lens.unsqueeze(1)    # (bs, max_len)
        attn_mask = valid.view(bs, 1, 1, max_len)

        q_ = q.unsqueeze(-2)             # (bs, H, 1, D)
        k_ = k.transpose(1, 2)           # (bs, Hkv, max_len, D)
        v_ = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
        )
        return out.squeeze(-2)           # (bs, H, D)
