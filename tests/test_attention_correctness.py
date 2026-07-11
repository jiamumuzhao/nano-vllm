import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_triton_varlen_prefill_matches_pytorch():
    torch.manual_seed(0)
    length, num_heads, num_kv_heads, head_dim = 19, 4, 2, 64
    q = torch.randn(length, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(length, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.k_cache = torch.empty(1, 256, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    attention.v_cache = torch.empty_like(attention.k_cache)
    try:
        set_context(True, torch.tensor([0, length], device="cuda", dtype=torch.int32), torch.tensor([0, length], device="cuda", dtype=torch.int32), slot_mapping=torch.arange(length, device="cuda", dtype=torch.int32))
        actual = attention(q, k, v)
        expected = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), is_causal=True, scale=attention.scale, enable_gqa=True).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        reset_context()


def test_triton_paged_decode_matches_pytorch():
    torch.manual_seed(1)
    length, num_heads, num_kv_heads, head_dim = 17, 4, 2, 64
    q = torch.randn(length, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(length, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.k_cache = torch.zeros(1, 256, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    attention.v_cache = torch.zeros_like(attention.k_cache)
    attention.k_cache[0, :length], attention.v_cache[0, :length] = k, v
    try:
        set_context(False, slot_mapping=torch.tensor([length - 1], device="cuda", dtype=torch.int32), context_lens=torch.tensor([length], device="cuda", dtype=torch.int32), block_tables=torch.tensor([[0]], device="cuda", dtype=torch.int32))
        actual = attention(q[-1:], k[-1:], v[-1:])
        expected = F.scaled_dot_product_attention(q[-1:].transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), scale=attention.scale, enable_gqa=True).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        reset_context()


def test_triton_paged_prefill_matches_reference():
    torch.manual_seed(2)
    prefix_len, new_len, num_heads, num_kv_heads, head_dim = 260, 17, 4, 2, 64
    total_len = prefix_len + new_len
    prefix_k = torch.randn(prefix_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    q = torch.randn(new_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k_new = torch.randn(new_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    k_full = torch.cat([prefix_k, k_new], dim=0)
    v_full = torch.cat([prefix_v, v_new], dim=0)
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.k_cache = torch.empty(2, 256, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    attention.v_cache = torch.empty_like(attention.k_cache)
    attention.k_cache.view(-1, num_kv_heads, head_dim)[:prefix_len] = prefix_k
    attention.v_cache.view(-1, num_kv_heads, head_dim)[:prefix_len] = prefix_v
    slot_mapping = torch.arange(prefix_len, total_len, device="cuda", dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    try:
        set_context(
            True,
            torch.tensor([0, new_len], device="cuda", dtype=torch.int32),
            torch.tensor([0, total_len], device="cuda", dtype=torch.int32),
            max_seqlen_q=new_len,
            max_seqlen_k=total_len,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        actual = attention(q, k_new, v_new)
        k_by_head = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1).float()
        v_by_head = v_full.repeat_interleave(num_heads // num_kv_heads, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q.float(), k_by_head) * attention.scale
        allowed = torch.arange(total_len, device="cuda").unsqueeze(0) <= (
            prefix_len + torch.arange(new_len, device="cuda").unsqueeze(1)
        )
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        expected = torch.einsum("hqk,khd->qhd", probs, v_by_head).to(torch.float16)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        reset_context()
