"""Benchmark eager paged decode with and without Split-KV.

Split-KV is intentionally not benchmarked under CUDA Graphs because the current
implementation disables it during graph capture.
"""
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
from nanovllm.layers.attention import Attention, SPLIT_KV_BUCKETS
from nanovllm.utils.context import reset_context, set_context

BLOCK_SIZE = 16
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64


def _ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    return float(torch.tensor(values, dtype=torch.float64).quantile(percentile / 100).item())


def _partition_bucket(context_len: int, enabled: bool, threshold: int,
                      partition_size: int, max_partitions: int) -> tuple[int, str | None]:
    if not enabled or context_len < threshold:
        return 0, None
    requested = max(1, (context_len + partition_size - 1) // partition_size)
    for bucket in SPLIT_KV_BUCKETS:
        if requested <= bucket:
            if bucket > max_partitions:
                break
            return bucket, None
    return 0, (
        f"context_len={context_len} requires a Split-KV bucket for {requested} partitions, "
        f"exceeding max_partitions={max_partitions}"
    )


def _make_inputs(batch_size: int, context_len: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    blocks_per_seq = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = batch_size * blocks_per_seq
    k_cache = torch.randn(total_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(total_blocks, device="cuda", dtype=torch.int32).reshape(
        batch_size, blocks_per_seq
    )
    context_lens = torch.full((batch_size,), context_len, device="cuda", dtype=torch.int32)
    q = torch.randn(batch_size, HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    k_new = torch.randn(batch_size, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    slots = block_tables[:, -1] * BLOCK_SIZE + (context_len - 1) % BLOCK_SIZE
    return dict(k_cache=k_cache, v_cache=v_cache, block_tables=block_tables,
                context_lens=context_lens, q=q, k_new=k_new, v_new=v_new,
                slots=slots)


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def benchmark_case(batch_size: int, context_len: int, split_kv_enabled: bool,
                   iters: int, warmup_iters: int, threshold: int,
                   partition_size: int, max_partitions: int, seed: int) -> dict:
    bucket, unsupported_reason = _partition_bucket(
        context_len, split_kv_enabled, threshold, partition_size, max_partitions
    )
    mode = "split_kv" if split_kv_enabled else "baseline"
    result = {
        "batch_size": batch_size,
        "context_len": context_len,
        "split_kv_enabled": split_kv_enabled,
        "mode": mode,
        "status": "unsupported" if unsupported_reason else "ok",
        "unsupported_reason": unsupported_reason,
        "partition_bucket": bucket,
        "step_latencies_ms": [],
        "mean_step_ms": None,
        "p50_step_ms": None,
        "p95_step_ms": None,
        "decode_tok_s": None,
        "peak_memory_allocated_bytes": None,
        "workspace_bytes": 0,
        "seed": seed,
    }
    if unsupported_reason:
        return result

    inputs = _make_inputs(batch_size, context_len, seed)
    config = dict(
        split_kv_enabled=split_kv_enabled,
        split_kv_threshold=threshold,
        split_kv_partition_size=partition_size,
        split_kv_max_partitions=max_partitions,
    )
    attention = Attention(HEADS, HEAD_DIM, HEAD_DIM ** -0.5, KV_HEADS,
                          split_kv_config=config)
    attention.initialize_split_workspace(batch_size, torch.device("cuda"))
    result["workspace_bytes"] = sum(x.numel() * x.element_size() for x in attention._split_workspace)
    attention.k_cache = inputs["k_cache"]
    attention.v_cache = inputs["v_cache"]

    try:
        with torch.inference_mode():
            set_context(False, slot_mapping=inputs["slots"],
                        context_lens=inputs["context_lens"],
                        block_tables=inputs["block_tables"])
            for _ in range(warmup_iters):
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
        result["step_latencies_ms"] = latencies
        result["mean_step_ms"] = statistics.fmean(latencies)
        result["p50_step_ms"] = _percentile(latencies, 50)
        result["p95_step_ms"] = _percentile(latencies, 95)
        result["decode_tok_s"] = batch_size * 1000.0 / result["mean_step_ms"]
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    finally:
        reset_context()
    return result


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(data: dict) -> str:
    rows = []
    grouped = {}
    for result in data["results"]:
        grouped.setdefault((result["context_len"], result["batch_size"]), {})[result["mode"]] = result
    rows.append("# Nano-vLLM eager paged decode: baseline vs Split-KV")
    rows.append("")
    rows.append("Split-KV is eager-only; CUDA Graph comparison is intentionally omitted.")
    rows.append("")
    rows.append("| Context | Batch | Mode | Mean ms | P50 ms | P95 ms | Decode tok/s | Speedup | Bucket | Peak MB | Status |")
    rows.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for (context_len, batch_size), modes in sorted(grouped.items()):
        baseline = modes.get("baseline")
        baseline_mean = baseline.get("mean_step_ms") if baseline else None
        for mode in ("baseline", "split_kv"):
            result = modes.get(mode)
            if result is None:
                continue
            speedup = None
            if mode == "split_kv" and baseline_mean and result["mean_step_ms"]:
                speedup = baseline_mean / result["mean_step_ms"]
            peak_mb = (result["peak_memory_allocated_bytes"] / 1024**2
                       if result["peak_memory_allocated_bytes"] is not None else None)
            rows.append("| {context_len} | {batch_size} | {mode} | {mean} | {p50} | {p95} | {tok} | {speed} | {bucket} | {peak} | {status} |".format(
                context_len=context_len, batch_size=batch_size, mode=mode,
                mean=_fmt(result["mean_step_ms"]), p50=_fmt(result["p50_step_ms"]),
                p95=_fmt(result["p95_step_ms"]), tok=_fmt(result["decode_tok_s"], 1),
                speed=_fmt(speedup, 3), bucket=result["partition_bucket"],
                peak=_fmt(peak_mb, 1), status=result["status"],
            ))
            if result["unsupported_reason"]:
                rows.append(f"| | | reason | {result['unsupported_reason']} | | | | | | | |")
    rows.extend(["", f"Generated: {data['environment']['timestamp_utc']}", ""])
    return "\n".join(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_ints, default=[1, 2, 4])
    parser.add_argument("--context-lens", type=_ints, default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--partition-size", type=int, default=1024)
    parser.add_argument("--max-partitions", type=int, default=16)
    parser.add_argument("--threshold", type=int, default=1024)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--split-kv", choices=("off", "on", "both"), default="both")
    parser.add_argument("--smoke", action="store_true", help="run a tiny both-mode smoke benchmark")
    args = parser.parse_args(argv)
    if args.iters < 1 or args.warmup_iters < 0:
        parser.error("--iters must be positive and --warmup-iters must be non-negative")
    if args.partition_size < 1 or args.max_partitions < 1 or args.threshold < 1:
        parser.error("partition size, max partitions, and threshold must be positive")
    if args.smoke:
        args.batch_sizes, args.context_lens = [1], [1024, 2048]
        args.iters, args.warmup_iters, args.split_kv = 1, 1, "both"
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this eager decode benchmark")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    modes = (False, True) if args.split_kv == "both" else (args.split_kv == "on",)
    results = []
    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            for enabled in modes:
                seed = 20260715 + batch_size * 100000 + context_len
                results.append(benchmark_case(
                    batch_size, context_len, enabled, args.iters, args.warmup_iters,
                    args.threshold, args.partition_size, args.max_partitions, seed,
                ))
                result = results[-1]
                print(f"{result['mode']:>9} batch={batch_size:>2} context={context_len:>5} "
                      f"status={result['status']} mean_ms={_fmt(result['mean_step_ms'])} "
                      f"tok/s={_fmt(result['decode_tok_s'], 1)} bucket={result['partition_bucket']}", flush=True)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device": torch.cuda.get_device_properties(0).name,
        "mode": "eager_only",
        "cuda_graph_split_kv": "disabled",
    }
    data = {
        "environment": environment,
        "command": " ".join(sys.argv),
        "parameters": vars(args),
        "benchmark_constants": {"block_size": BLOCK_SIZE, "heads": HEADS,
                                 "kv_heads": KV_HEADS, "head_dim": HEAD_DIM},
        "results": results,
    }
    json_path = output_dir / f"decode_split_kv_{timestamp}.json"
    markdown_path = output_dir / f"decode_split_kv_{timestamp}.md"
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    markdown_path.write_text(render_markdown(data))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
