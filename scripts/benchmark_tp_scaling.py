#!/usr/bin/env python3
"""Run identical online serving workloads across tensor-parallel sizes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_sizes(raw):
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("tensor-parallel-sizes must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("tensor-parallel-sizes must not contain duplicates")
    return values


def load_records(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") == "serving_summary" and not record.get("warmup"):
            records.append(record)
    return records


def average(records, field, nested=None):
    values = []
    for record in records:
        if record.get("status") != "ok":
            continue
        value = record
        if nested:
            for key in nested:
                value = value.get(key) if isinstance(value, dict) else None
        value = value.get(field) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def summarize(size, path):
    records = load_records(path)
    successful = [record for record in records if record.get("status") == "ok"]
    return {
        "tensor_parallel_size": size,
        "jsonl": str(path),
        "records": len(records),
        "successful_records": len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "output_tokens_per_second": average(records, "global_output_tokens_per_second"),
        "ttft_p95": average(records, "p95", ["ttft_s"]),
        "e2e_p95": average(records, "p95", ["e2e_latency_s"]),
    }


def render(rows):
    baseline = next((row for row in rows if row["tensor_parallel_size"] == 1), None)
    lines = [
        "# Tensor Parallel Scaling Benchmark",
        "",
        "| TP size | records | success | output tok/s | TTFT p95 (s) | E2E p95 (s) | scaling efficiency |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        efficiency = None
        if baseline and baseline["output_tokens_per_second"] and row["output_tokens_per_second"]:
            efficiency = row["output_tokens_per_second"] / (
                baseline["output_tokens_per_second"] * row["tensor_parallel_size"]
            )
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['tensor_parallel_size']} | {row['records']} | "
            f"{row['success_rate']:.2%} | {fmt(row['output_tokens_per_second'])} | "
            f"{fmt(row['ttft_p95'])} | {fmt(row['e2e_p95'])} | {fmt(efficiency)} |"
        )
    lines += [
        "",
        "Scaling efficiency is TP-N throughput divided by TP-N times TP-1 throughput.",
        "Compare only runs with identical model, workload, dtype, GPU type, and scheduler parameters.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-sizes", default="1,2")
    parser.add_argument("--output-dir", default="docs/benchmarks/tp_scaling")
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="64")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--scheduling-policy", default="throughput")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    sizes = parse_sizes(args.tensor_parallel_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sizes:
        jsonl = output_dir / f"tp{size}.jsonl"
        markdown = output_dir / f"tp{size}.md"
        command = [
            sys.executable, "scripts/benchmark_serving.py",
            "--model", args.model,
            "--tensor-parallel-size", str(size),
            "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--dtype", args.dtype,
            "--runs", str(args.runs),
            "--warmup-runs", str(args.warmup_runs),
            "--scheduling-policy", args.scheduling_policy,
            "--concurrencies", args.concurrencies,
            "--input-lens", args.input_lens,
            "--output-lens", args.output_lens,
            "--output-jsonl", str(jsonl),
            "--output-md", str(markdown),
        ]
        if args.enforce_eager:
            command.append("--enforce-eager")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"TP={size} benchmark failed with exit code {completed.returncode}")
        rows.append(summarize(size, jsonl))
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(rows, indent=2) + "\n")
    report = output_dir / "summary.md"
    report.write_text(render(rows) + "\n")
    print(report)


if __name__ == "__main__":
    raise SystemExit(main())
