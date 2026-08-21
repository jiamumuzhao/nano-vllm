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


@pytest.mark.parametrize("length", [1025, 8193, 16384])
def test_triton_split_paged_decode_matches_pytorch(length):
    torch.manual_seed(length)
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 64, 256
    lengths = [length, max(777, length // 2)]
    num_blocks = [(n + block_size - 1) // block_size for n in lengths]
    max_blocks = max(num_blocks)
    k_cache = torch.zeros(sum(num_blocks), block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    block_tables, keys, values = [], [], []
    block_base = 0
    for n, blocks in zip(lengths, num_blocks):
        key = torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        ids = list(range(block_base, block_base + blocks))
        block_base += blocks
        block_tables.append(ids + [-1] * (max_blocks - blocks))
        keys.append(key)
        values.append(value)
        for block_idx, block_id in enumerate(ids):
            start = block_idx * block_size
            k_cache[block_id, :n - start] = key[start:start + block_size]
            v_cache[block_id, :n - start] = value[start:start + block_size]

    q = torch.randn(len(lengths), num_heads, head_dim, device="cuda", dtype=torch.float16)
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.initialize_split_workspace(len(lengths), torch.device("cuda"), max_partitions=16)
    workspace_ptrs = tuple(x.data_ptr() for x in attention._split_workspace)
    attention.k_cache, attention.v_cache = k_cache, v_cache
    try:
        set_context(
            False,
            slot_mapping=torch.tensor([n - 1 for n in lengths], device="cuda", dtype=torch.int32),
            context_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            block_tables=torch.tensor(block_tables, device="cuda", dtype=torch.int32),
        )
        actual = attention(q, torch.stack([key[-1] for key in keys]), torch.stack([value[-1] for value in values]))
        expected = []
        for i, key in enumerate(keys):
            expected.append(F.scaled_dot_product_attention(
                q[i:i + 1].transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                values[i].transpose(0, 1).unsqueeze(0),
                scale=attention.scale, enable_gqa=True,
            ).squeeze(0).transpose(0, 1).squeeze(0))
        torch.testing.assert_close(actual, torch.stack(expected).to(actual.dtype), rtol=2e-2, atol=2e-2)
        assert workspace_ptrs == tuple(x.data_ptr() for x in attention._split_workspace)
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


def test_split_kv_mixed_batch_lengths():
    torch.manual_seed(4097)
    lengths = [4097, 1024, 17]
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 64, 256
    blocks = [(n + block_size - 1) // block_size for n in lengths]
    max_blocks = max(blocks)
    k_cache = torch.zeros(sum(blocks), block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    rows, keys, values = [], [], []
    base = 0
    for n, count in zip(lengths, blocks):
        key = torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key) * 0.25
        ids = list(range(base, base + count)); base += count
        rows.append(ids + [-1] * (max_blocks - count))
        keys.append(key); values.append(value)
        for i, block_id in enumerate(ids):
            valid = min(block_size, n - i * block_size)
            k_cache[block_id, :valid] = key[i * block_size:i * block_size + valid]
            v_cache[block_id, :valid] = value[i * block_size:i * block_size + valid]
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.initialize_split_workspace(len(lengths), torch.device("cuda"), max_partitions=16)
    attention.k_cache, attention.v_cache = k_cache, v_cache
    try:
        set_context(False, slot_mapping=torch.tensor([n - 1 for n in lengths], device="cuda", dtype=torch.int32),
                    context_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    block_tables=torch.tensor(rows, device="cuda", dtype=torch.int32))
        q = torch.randn(len(lengths), num_heads, head_dim, device="cuda", dtype=torch.float16)
        actual = attention(q, torch.stack([x[-1] for x in keys]), torch.stack([x[-1] for x in values]))
        expected = []
        for i, (key, value) in enumerate(zip(keys, values)):
            expected.append(F.scaled_dot_product_attention(
                q[i:i + 1].float().transpose(0, 1).unsqueeze(0),
                key.float().transpose(0, 1).unsqueeze(0),
                value.float().transpose(0, 1).unsqueeze(0),
                scale=attention.scale, enable_gqa=True,
            ).squeeze(0).transpose(0, 1).squeeze(0))
        torch.testing.assert_close(actual, torch.stack(expected).to(actual.dtype), rtol=2e-2, atol=2e-2)
    finally:
        reset_context()


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_split_kv_head_dims_and_mha(head_dim):
    torch.manual_seed(head_dim)
    n, num_heads, num_kv_heads, block_size = 1025, 4, 4, 256
    blocks = (n + block_size - 1) // block_size
    key = torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    k_cache = torch.zeros(blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, num_kv_heads, head_dim)[:n] = key
    v_cache.view(-1, num_kv_heads, head_dim)[:n] = value
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.initialize_split_workspace(1, torch.device("cuda"), max_partitions=16)
    attention.k_cache, attention.v_cache = k_cache, v_cache
    try:
        set_context(False, slot_mapping=torch.tensor([n - 1], device="cuda", dtype=torch.int32),
                    context_lens=torch.tensor([n], device="cuda", dtype=torch.int32),
                    block_tables=torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1))
        q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
        actual = attention(q, key[-1:], value[-1:])
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            scale=attention.scale, enable_gqa=False,
        ).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        reset_context()


def test_split_kv_repeated_decode_keeps_workspace():
    torch.manual_seed(1234)
    lengths = [1025, 4097, 8193]
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 64, 256
    max_len = max(lengths)
    max_blocks = (max_len + block_size - 1) // block_size
    key = torch.randn(max_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    k_cache = torch.zeros(max_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, num_kv_heads, head_dim)[:max_len] = key
    v_cache.view(-1, num_kv_heads, head_dim)[:max_len] = value
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads)
    attention.initialize_split_workspace(1, torch.device("cuda"), max_partitions=16)
    workspace_ptrs = tuple(x.data_ptr() for x in attention._split_workspace)
    attention.k_cache, attention.v_cache = k_cache, v_cache
    try:
        for n in lengths:
            blocks = (n + block_size - 1) // block_size
            q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
            set_context(False,
                        slot_mapping=torch.tensor([n - 1], device="cuda", dtype=torch.int32),
                        context_lens=torch.tensor([n], device="cuda", dtype=torch.int32),
                        block_tables=torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1))
            actual = attention(q, key[n - 1:n], value[n - 1:n])
            expected = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                key[:n].transpose(0, 1).unsqueeze(0),
                value[:n].transpose(0, 1).unsqueeze(0),
                scale=attention.scale, enable_gqa=True,
            ).squeeze(0).transpose(0, 1)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert workspace_ptrs == tuple(x.data_ptr() for x in attention._split_workspace)
            reset_context()
    finally:
        reset_context()



def _run_paged_prefill_q_tile_case(mode: str):
    torch.manual_seed(31415)
    prefix_lens = [32, 48]
    new_lens = [17, 65]
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 64, 16
    full_lens = [p + n for p, n in zip(prefix_lens, new_lens)]
    block_counts = [(n + block_size - 1) // block_size for n in full_lens]
    max_blocks = max(block_counts)
    total_blocks = sum(block_counts)
    prefixes = [torch.randn(p, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
                for p in prefix_lens]
    prefix_values = [torch.randn_like(x) for x in prefixes]
    new_keys = [torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
                for n in new_lens]
    new_values = [torch.randn_like(x) for x in new_keys]
    q_parts = [torch.randn(n, num_heads, head_dim, device="cuda", dtype=torch.float16)
               for n in new_lens]
    block_tables_rows, slot_mapping = [], []
    k_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    block_base = 0
    for prefix, prefix_value, full_len, count in zip(prefixes, prefix_values, full_lens, block_counts):
        ids = list(range(block_base, block_base + count))
        block_base += count
        block_tables_rows.append(ids + [-1] * (max_blocks - count))
        for block_idx, block_id in enumerate(ids):
            start = block_idx * block_size
            valid = min(block_size, max(0, len(prefix) - start))
            if valid:
                k_cache[block_id, :valid] = prefix[start:start + valid]
                v_cache[block_id, :valid] = prefix_value[start:start + valid]
        for token_idx in range(len(prefix), full_len):
            block_idx, offset = divmod(token_idx, block_size)
            slot_mapping.append(ids[block_idx] * block_size + offset)

    q = torch.cat(q_parts)
    k_new = torch.cat(new_keys)
    v_new = torch.cat(new_values)
    cu_q = torch.tensor([0, new_lens[0], sum(new_lens)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, full_lens[0], sum(full_lens)], device="cuda", dtype=torch.int32)
    context_lens_q = max(new_lens)
    attention = Attention(num_heads, head_dim, head_dim**-0.5, num_kv_heads,
                          split_kv_config={"paged_prefill_q_tile_mode": mode})
    attention.k_cache = k_cache
    attention.v_cache = v_cache
    try:
        set_context(True, cu_q, cu_k, context_lens_q, max(full_lens),
                    torch.tensor(slot_mapping, device="cuda", dtype=torch.int32),
                    None, torch.tensor(block_tables_rows, device="cuda", dtype=torch.int32))
        actual = attention(q, k_new, v_new)
        expected_parts = []
        for qi, prefix, prefix_value, knew, vnew, q_len, prefix_len in zip(
                q_parts, prefixes, prefix_values, new_keys, new_values, new_lens, prefix_lens):
            k_full = torch.cat([prefix, knew])
            v_full = torch.cat([prefix_value, vnew])
            allowed = torch.arange(k_full.size(0), device="cuda")[None, :] <= (
                prefix_len + torch.arange(q_len, device="cuda")[:, None]
            )
            expected_parts.append(F.scaled_dot_product_attention(
                qi.transpose(0, 1).unsqueeze(0),
                k_full.transpose(0, 1).unsqueeze(0),
                v_full.transpose(0, 1).unsqueeze(0),
                attn_mask=allowed.unsqueeze(0).unsqueeze(0),
                scale=attention.scale, enable_gqa=True,
            ).squeeze(0).transpose(0, 1))
        expected = torch.cat(expected_parts)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        return actual
    finally:
        reset_context()


def test_paged_prefill_q_tile_parallel_matches_serial_and_reference():
    serial = _run_paged_prefill_q_tile_case("serial")
    parallel = _run_paged_prefill_q_tile_case("parallel")
    auto = _run_paged_prefill_q_tile_case("auto")
    torch.testing.assert_close(parallel, serial, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(auto, serial, rtol=2e-2, atol=2e-2)


def test_split_kv_partition_limit_raises():
    attention = Attention(4, 64, 64**-0.5, 2)
    with pytest.raises(ValueError, match="exceeding split_kv_max_partitions"):
        attention._split_partition_count(16385)
