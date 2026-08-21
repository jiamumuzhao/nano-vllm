"""Eager benchmark for paged-prefill Q-tile serial vs parallel paths."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context

BLOCK_SIZE = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64


def _ints(value):
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _percentile(values, p):
    return float(torch.tensor(values, dtype=torch.float64).quantile(p / 100).item())


def make_case(batch, prefix_len, new_tokens, seed):
    torch.manual_seed(seed)
    total_len = prefix_len + new_tokens
    blocks_per_seq = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = batch * blocks_per_seq
    prefix = torch.randn(batch * prefix_len, NUM_KV_HEADS, HEAD_DIM,
                         device="cuda", dtype=torch.float16)
    prefix_v = torch.randn_like(prefix)
    q = torch.randn(batch * new_tokens, NUM_HEADS, HEAD_DIM,
                    device="cuda", dtype=torch.float16)
    k_new = torch.randn(batch * new_tokens, NUM_KV_HEADS, HEAD_DIM,
                        device="cuda", dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    k_cache = torch.zeros(total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    rows, slots = [], []
    for b in range(batch):
        base = b * blocks_per_seq
        ids = list(range(base, base + blocks_per_seq))
        rows.append(ids)
        prefix_start = b * prefix_len
        for token in range(prefix_len):
            block, offset = divmod(token, BLOCK_SIZE)
            k_cache[ids[block], offset] = prefix[prefix_start + token]
            v_cache[ids[block], offset] = prefix_v[prefix_start + token]
        for token in range(prefix_len, total_len):
            block, offset = divmod(token, BLOCK_SIZE)
            slots.append(ids[block] * BLOCK_SIZE + offset)
    cu_q = torch.arange(0, (batch + 1) * new_tokens, new_tokens,
                        device="cuda", dtype=torch.int32)
    cu_k = torch.arange(0, (batch + 1) * total_len, total_len,
                        device="cuda", dtype=torch.int32)
    return dict(q=q, k_new=k_new, v_new=v_new, k_cache=k_cache,
                v_cache=v_cache, block_tables=torch.tensor(rows, device="cuda", dtype=torch.int32),
                slots=torch.tensor(slots, device="cuda", dtype=torch.int32),
                cu_q=cu_q, cu_k=cu_k, max_q=new_tokens, max_k=total_len)


def run_case(inputs, batch, prefix_len, new_tokens, parallel, block_q, block_kv, num_warps, iters, warmups):
    attention = Attention(NUM_HEADS, HEAD_DIM, HEAD_DIM ** -0.5, NUM_KV_HEADS,
                          split_kv_config={"paged_prefill_q_tile_parallel": parallel, "paged_prefill_block_q": block_q, "paged_prefill_block_kv": block_kv, "paged_prefill_num_warps": num_warps})
    attention.k_cache = inputs["k_cache"].clone()
    attention.v_cache = inputs["v_cache"].clone()
    try:
        with torch.inference_mode():
            set_context(True, inputs["cu_q"], inputs["cu_k"], inputs["max_q"],
                        inputs["max_k"], inputs["slots"], None, inputs["block_tables"])
            for _ in range(warmups):
                attention(inputs["q"], inputs["k_new"], inputs["v_new"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            latencies = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                attention(inputs["q"], inputs["k_new"], inputs["v_new"])
                end.record()
                end.synchronize()
                latencies.append(float(start.elapsed_time(end)))
        mean = statistics.fmean(latencies)
        return {
            "batch_size": batch, "prefix_len": prefix_len, "new_tokens": new_tokens,
            "parallel": parallel, "status": "ok", "step_latencies_ms": latencies,
            "mean_step_ms": mean, "p50_step_ms": _percentile(latencies, 50),
            "p95_step_ms": _percentile(latencies, 95),
            "prefill_tok_s": batch * new_tokens * 1000.0 / mean,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "block_q": block_q, "block_kv": block_kv, "num_warps": num_warps,
        }
    finally:
        reset_context()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_ints, default=[1, 4, 8])
    parser.add_argument("--prefix-lengths", type=_ints, default=[1024, 2048, 4096])
    parser.add_argument("--new-tokens", type=_ints, default=[16, 64, 256, 512])
    parser.add_argument("--block-qs", type=_ints, default=[16, 32, 64])
    parser.add_argument("--block-kvs", type=_ints, default=[64, 128])
    parser.add_argument("--num-warps", type=_ints, default=[4, 8])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    return parser.parse_args(argv)


def markdown(data):
    grouped = {}
    for result in data["results"]:
        key = (result["prefix_len"], result["new_tokens"], result["batch_size"],
               result["block_q"], result["block_kv"], result["num_warps"])
        grouped.setdefault(key, {})[result["parallel"]] = result
    lines = ["# Nano-vLLM paged prefill Q-tile parameter sweep", "",
             "Eager-only comparison; CUDA Graph is not used. Every parameter combination has its own serial and parallel result.", "",
             "| Prefix | New | Batch | Path | Mean ms | P50 ms | P95 ms | Prefill tok/s | Speedup vs serial | Peak MB | BLOCK_Q | BLOCK_KV | Warps | Status |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, paths in sorted(grouped.items()):
        serial = paths.get(False)
        serial_mean = serial["mean_step_ms"] if serial else None
        for parallel in (False, True):
            r = paths.get(parallel)
            if not r:
                continue
            speedup = serial_mean / r["mean_step_ms"] if parallel and serial_mean else None
            peak = r["peak_memory_allocated_bytes"] / 1024**2
            lines.append("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {} | {:.1f} | {} | {} | {} | {} |".format(
                r["prefix_len"], r["new_tokens"], r["batch_size"],
                "parallel" if parallel else "serial", r["mean_step_ms"], r["p50_step_ms"],
                r["p95_step_ms"], r["prefill_tok_s"],
                "—" if speedup is None else f"{speedup:.3f}", peak,
                r["block_q"], r["block_kv"], r["num_warps"], r["status"]))
    lines += ["", "## Best parallel configuration per workload", "",
              "| Prefix | New | Batch | BLOCK_Q | BLOCK_KV | Warps | Parallel mean ms | Speedup vs best serial baseline |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    workloads = {}
    for r in data["results"]:
        workloads.setdefault((r["prefix_len"], r["new_tokens"], r["batch_size"], r["parallel"]), []).append(r)
    for prefix, new, batch in sorted({k[:3] for k in workloads}):
        serials = workloads.get((prefix, new, batch, False), [])
        parallels = workloads.get((prefix, new, batch, True), [])
        if not serials or not parallels:
            continue
        best_serial = min(serials, key=lambda x: x["mean_step_ms"])
        best_parallel = min(parallels, key=lambda x: x["mean_step_ms"])
        speedup = best_serial["mean_step_ms"] / best_parallel["mean_step_ms"]
        lines.append(f"| {prefix} | {new} | {batch} | {best_parallel['block_q']} | {best_parallel['block_kv']} | {best_parallel['num_warps']} | {best_parallel['mean_step_ms']:.3f} | {speedup:.3f} |")
    regressions = []
    for key, paths in grouped.items():
        if paths.get(False) and paths.get(True):
            ratio = paths[True]["mean_step_ms"] / paths[False]["mean_step_ms"]
            if ratio > 1:
                regressions.append((ratio, key, paths[True]["mean_step_ms"], paths[False]["mean_step_ms"]))
    lines += ["", "## Parallel regressions", ""]
    if regressions:
        lines.append("The following parameter/workload combinations are slower in parallel mode; none are filtered from the raw table.")
        lines.append("")
        lines.append("| Prefix | New | Batch | BLOCK_Q | BLOCK_KV | Warps | Serial ms | Parallel ms | Parallel/serial |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ratio, key, parallel_ms, serial_ms in sorted(regressions, reverse=True):
            prefix, new, batch, block_q, block_kv, warps = key
            lines.append(f"| {prefix} | {new} | {batch} | {block_q} | {block_kv} | {warps} | {serial_ms:.3f} | {parallel_ms:.3f} | {ratio:.3f} |")
    else:
        lines.append("No parallel regression was observed in this run.")
    lines += ["", f"Generated: {data['environment']['timestamp_utc']}", ""]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in args.batch_sizes:
        for prefix_len in args.prefix_lengths:
            for new_tokens in args.new_tokens:
                seed = 20260715 + batch * 100000 + prefix_len * 10 + new_tokens
                inputs = make_case(batch, prefix_len, new_tokens, seed)
                for block_q in args.block_qs:
                    for block_kv in args.block_kvs:
                        for num_warps in args.num_warps:
                            for parallel in (False, True):
                                result = run_case(inputs, batch, prefix_len, new_tokens, parallel,
                                                  block_q, block_kv, num_warps,
                                                  args.iters, args.warmup_iters)
                                result.update(seed=seed, block_q=block_q, block_kv=block_kv,
                                              num_warps=num_warps)
                                results.append(result)
                                print(f"{'parallel' if parallel else 'serial':>8} batch={batch} prefix={prefix_len} new={new_tokens} q={block_q} kv={block_kv} warps={num_warps} mean_ms={result['mean_step_ms']:.3f}", flush=True)
    data = {
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "triton_version": triton.__version__,
            "mode": "eager_only",
            "cuda_graph": "not used",
            "block_size": BLOCK_SIZE,
        },
        "command": " ".join(sys.argv),
        "parameters": vars(args),
        "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"paged_prefill_q_tile_{stamp}.json"
    md_path = output_dir / f"paged_prefill_q_tile_{stamp}.md"
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    md_path.write_text(markdown(data))
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
