#!/usr/bin/env python3
"""Summarize online serving benchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def load_records(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") != "serving_summary" or record.get("warmup"):
            continue
        records.append(record)
    if not records:
        raise ValueError("no measured serving_summary records found")
    return records


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        key = (
            record.get("scheduling_policy", "fcfs"),
            record.get("preemption_cooldown_steps", 2),
            record.get("max_preemptions_per_step", 1),
        )
        groups[key].append(record)

    rows = []
    for (policy, cooldown, max_preempts), items in groups.items():
        successful = [item for item in items if item.get("status") == "ok"]
        throughput = mean([finite(item.get("global_output_tokens_per_second")) for item in successful])
        ttft_p95 = mean([finite((item.get("ttft_s") or {}).get("p95")) for item in successful])
        e2e_p95 = mean([finite((item.get("e2e_latency_s") or {}).get("p95")) for item in successful])
        scheduler = [(item.get("metrics") or {}).get("scheduler") or {} for item in successful]
        preemptions = mean([finite(item.get("preemption_count", 0)) for item in scheduler])
        recompute = mean([finite(item.get("preemption_recompute_tokens", 0)) for item in scheduler])
        deferred = mean([finite((item.get("metrics") or {}).get("deferred_requests", 0)) for item in successful])
        rows.append({
            "policy": policy,
            "cooldown": cooldown,
            "max_preempts": max_preempts,
            "runs": len(items),
            "success_rate": len(successful) / len(items),
            "throughput": throughput,
            "ttft_p95": ttft_p95,
            "e2e_p95": e2e_p95,
            "preemptions": preemptions,
            "recompute": recompute,
            "deferred": deferred,
        })

    def score(row):
        throughput = row["throughput"] or 0.0
        latency = row["e2e_p95"] or float("inf")
        return (row["success_rate"], throughput / max(latency, 1e-9))

    return sorted(rows, key=score, reverse=True)


def validate_thresholds(rows, args):
    failures = []
    for row in rows:
        label = (
            f"{row['policy']}/cooldown={row['cooldown']}/"
            f"max_preempts={row['max_preempts']}"
        )
        if args.min_success_rate is not None and row["success_rate"] < args.min_success_rate:
            failures.append(
                f"{label}: success rate {row['success_rate']:.4f} < "
                f"minimum {args.min_success_rate:.4f}"
            )
        if args.min_throughput is not None and (
            row["throughput"] is None or row["throughput"] < args.min_throughput
        ):
            failures.append(
                f"{label}: throughput {fmt(row['throughput'])} < "
                f"minimum {args.min_throughput:.4f}"
            )
        if args.max_ttft_p95 is not None and (
            row["ttft_p95"] is None or row["ttft_p95"] > args.max_ttft_p95
        ):
            failures.append(
                f"{label}: TTFT p95 {fmt(row['ttft_p95'])} > "
                f"maximum {args.max_ttft_p95:.4f}"
            )
        if args.max_e2e_p95 is not None and (
            row["e2e_p95"] is None or row["e2e_p95"] > args.max_e2e_p95
        ):
            failures.append(
                f"{label}: E2E p95 {fmt(row['e2e_p95'])} > "
                f"maximum {args.max_e2e_p95:.4f}"
            )
    return failures


def render(rows, input_path):
    lines = [
        "# Serving Benchmark Analysis",
        "",
        f"Input: {input_path}",
        "",
        "Rows are grouped by scheduling policy and preemption parameters. Latency and throughput are arithmetic means across measured workload records.",
        "",
        "| rank | policy | cooldown | max preempts | runs | success | output tok/s | TTFT p95 (s) | E2E p95 (s) | preemptions | recompute tokens | deferred |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {row['policy']} | {row['cooldown']} | {row['max_preempts']} | "
            f"{row['runs']} | {row['success_rate']:.2%} | {fmt(row['throughput'])} | "
            f"{fmt(row['ttft_p95'])} | {fmt(row['e2e_p95'])} | {fmt(row['preemptions'])} | "
            f"{fmt(row['recompute'])} | {fmt(row['deferred'])} |"
        )
    lines.append("")
    if rows:
        best = rows[0]
        lines.append(
            f"Recommended first candidate: {best['policy']} with cooldown "
            f"{best['cooldown']} and max preempts {best['max_preempts']}. "
            "Validate this candidate with repeated runs before making it the deployment default."
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--output-md")
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--min-throughput", type=float)
    parser.add_argument("--max-ttft-p95", type=float)
    parser.add_argument("--max-e2e-p95", type=float)
    args = parser.parse_args()
    for name in ("min_success_rate", "min_throughput", "max_ttft_p95", "max_e2e_p95"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be a finite non-negative number")
    rows = summarize(load_records(args.input_jsonl))
    failures = validate_thresholds(rows, args)
    output = render(rows, args.input_jsonl)
    if failures:
        output += "\n## Threshold failures\n\n"
        output += "\n".join(f"- {failure}" for failure in failures) + "\n"
    if args.output_md:
        Path(args.output_md).write_text(output)
    else:
        print(output, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
