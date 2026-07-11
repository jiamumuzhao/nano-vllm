"""Compare old dense-assemble prefix prefill with the paged prefill kernel."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.utils.context import get_context, reset_context, set_context


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_case(args, batch_size: int, prefix_len: int, new_len: int):
    torch.manual_seed(args.seed)
    block_size = args.block_size
    total_len = prefix_len + new_len
    blocks_per_seq = (total_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    q = torch.randn(batch_size * new_len, args.num_heads, args.head_dim, device="cuda", dtype=torch.float16)
    k_new = torch.randn(batch_size * new_len, args.num_kv_heads, args.head_dim, device="cuda", dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    k_cache = torch.empty(num_blocks, block_size, args.num_kv_heads, args.head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    block_tables = torch.empty(batch_size, blocks_per_seq, device="cuda", dtype=torch.int32)
    slot_mapping = torch.empty(batch_size * new_len, device="cuda", dtype=torch.int32)

    for seq_idx in range(batch_size):
        base_block = seq_idx * blocks_per_seq
        block_tables[seq_idx] = torch.arange(base_block, base_block + blocks_per_seq, device="cuda", dtype=torch.int32)
        prefix_k = torch.randn(prefix_len, args.num_kv_heads, args.head_dim, device="cuda", dtype=torch.float16)
        prefix_v = torch.randn_like(prefix_k)
        k_cache[base_block:base_block + blocks_per_seq].view(-1, args.num_kv_heads, args.head_dim)[:prefix_len] = prefix_k
        v_cache[base_block:base_block + blocks_per_seq].view(-1, args.num_kv_heads, args.head_dim)[:prefix_len] = prefix_v
        for token_idx in range(new_len):
            logical = prefix_len + token_idx
            slot_mapping[seq_idx * new_len + token_idx] = base_block * block_size + logical

    cu_q = torch.arange(0, (batch_size + 1) * new_len, new_len, device="cuda", dtype=torch.int32)
    cu_k = torch.arange(0, (batch_size + 1) * total_len, total_len, device="cuda", dtype=torch.int32)
    attention = Attention(args.num_heads, args.head_dim, args.head_dim**-0.5, args.num_kv_heads)
    attention.k_cache = k_cache
    attention.v_cache = v_cache
    return attention, q, k_new, v_new, cu_q, cu_k, slot_mapping, block_tables


def old_prefill_paged(attention: Attention, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor):
    context = get_context()
    k_cache, v_cache = attention.k_cache, attention.v_cache
    store_kvcache(k_new, v_new, k_cache, v_cache, context.slot_mapping)
    cu_q, cu_k = context.cu_seqlens_q, context.cu_seqlens_k
    block_size = k_cache.size(1)
    batch_size = cu_q.numel() - 1
    total_k_len = cu_k[-1].item()
    k_full = torch.empty(total_k_len, *k_new.shape[1:], dtype=k_new.dtype, device=k_new.device)
    v_full = torch.empty(total_k_len, *v_new.shape[1:], dtype=v_new.dtype, device=v_new.device)
    block_tables = context.block_tables

    for i in range(batch_size):
        ks = int(cu_k[i])
        ke = int(cu_k[i + 1])
        qs = int(cu_q[i])
        qe = int(cu_q[i + 1])
        prefix_len = (ke - ks) - (qe - qs)
        if prefix_len > 0:
            nblk = (prefix_len + block_size - 1) // block_size
            bt = block_tables[i, :nblk].clamp(min=0)
            k_full[ks:ks + prefix_len] = k_cache[bt].reshape(-1, *k_new.shape[1:])[:prefix_len]
            v_full[ks:ks + prefix_len] = v_cache[bt].reshape(-1, *v_new.shape[1:])[:prefix_len]
            k_full[ks + prefix_len:ke] = k_new[qs:qe]
            v_full[ks + prefix_len:ke] = v_new[qs:qe]
        else:
            k_full[ks:ke] = k_new[qs:qe]
            v_full[ks:ke] = v_new[qs:qe]
    return attention._prefill_varlen(q, k_full, v_full, context)


def time_fn(fn, warmup_runs: int, runs: int) -> list[float]:
    for _ in range(warmup_runs):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return timings


def run_case(args, batch_size: int, prefix_len: int, new_len: int) -> dict:
    attention, q, k_new, v_new, cu_q, cu_k, slot_mapping, block_tables = build_case(args, batch_size, prefix_len, new_len)
    set_context(True, cu_q, cu_k, max_seqlen_q=new_len, max_seqlen_k=prefix_len + new_len, slot_mapping=slot_mapping, block_tables=block_tables)
    try:
        old_times = time_fn(lambda: old_prefill_paged(attention, q, k_new, v_new), args.warmup_runs, args.runs)
        new_times = time_fn(lambda: attention(q, k_new, v_new), args.warmup_runs, args.runs)
        old_ms = sum(old_times) / len(old_times) * 1000
        new_ms = sum(new_times) / len(new_times) * 1000
        return {
            "batch_size": batch_size,
            "prefix_len": prefix_len,
            "new_len": new_len,
            "old_dense_assemble_ms": old_ms,
            "new_paged_kernel_ms": new_ms,
            "speedup": old_ms / new_ms,
        }
    finally:
        reset_context()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--prefix-lens", default="1024")
    parser.add_argument("--new-lens", default="16")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = []
    for prefix_len in parse_csv_ints(args.prefix_lens):
        for new_len in parse_csv_ints(args.new_lens):
            for batch_size in parse_csv_ints(args.batch_sizes):
                result = run_case(args, batch_size, prefix_len, new_len)
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
