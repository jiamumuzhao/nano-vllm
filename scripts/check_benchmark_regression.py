"""Offline JSONL performance regression gate; never runs a model or touches CUDA."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_schema import read_jsonl_records


IDENTITY_FIELDS = (
    "engine", "model", "dtype", "tensor_parallel_size", "num_seqs", "input_len",
    "output_len", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "gpu_memory_utilization", "temperature",
)
ENVIRONMENT_FIELDS = ("gpu_names", "gpu_count", "cuda_runtime", "torch")


def nonnegative_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def lookup(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    workload = record.get("workload")
    if isinstance(workload, dict):
        return workload.get(field)
    return None


def identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(lookup(record, field) for field in IDENTITY_FIELDS)


def identity_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(IDENTITY_FIELDS, key))


def gpu_names(environment: dict[str, Any]) -> list[str]:
    return [str(item.get("name")) for item in environment.get("gpus", []) or []]


def environment(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("environment")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "gpu_names": gpu_names(raw),
        "gpu_count": raw.get("gpu_count"),
        "cuda_runtime": raw.get("cuda_runtime"),
        "torch": raw.get("torch"),
    }


def measured_runs(record: dict[str, Any]) -> tuple[list[dict[str, float]], list[str]]:
    """Validate every measured run; never silently discard an invalid run."""
    measurements = record.get("measurements")
    if not isinstance(measurements, list):
        return [], ["measurements must be a list"]
    result: list[dict[str, float]] = []
    errors: list[str] = []
    for position, item in enumerate(measurements):
        run_index = item.get("run_index", position) if isinstance(item, dict) else position
        if not isinstance(item, dict):
            errors.append(f"run index {run_index}: measurement must be a dict, got {item!r}")
            continue
        values: dict[str, float] = {}
        for field in ("output_tokens_per_second", "seconds"):
            if field not in item:
                errors.append(f"run index {run_index}: missing {field}")
                continue
            raw_value = item[field]
            if isinstance(raw_value, str):
                errors.append(f"run index {run_index}: {field}={raw_value!r} must be numeric; strings are invalid")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                errors.append(f"run index {run_index}: {field}={raw_value!r} is not float-convertible")
                continue
            if not math.isfinite(value) or value <= 0:
                errors.append(f"run index {run_index}: {field}={raw_value!r} must be finite and > 0")
                continue
            values[field] = value
        if len(values) == 2:
            result.append(values)
    return result, errors


def median_metrics(record: dict[str, Any]) -> dict[str, Any]:
    runs, errors = measured_runs(record)
    return {
        "run_count": len(record.get("measurements", [])) if isinstance(record.get("measurements"), list) else 0,
        "valid_run_count": len(runs),
        "median_output_tokens_per_second": statistics.median([r["output_tokens_per_second"] for r in runs]) if runs and not errors else None,
        "median_seconds": statistics.median([r["seconds"] for r in runs]) if runs and not errors else None,
        "measurement_errors": errors,
    }


def percent_change(candidate: float, baseline: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(candidate) or candidate <= 0:
        raise ValueError(f"percent_change requires finite positive values, baseline={baseline!r}, candidate={candidate!r}")
    return (candidate - baseline) / baseline * 100.0


def compare_records(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    min_runs: int,
    max_throughput_regression_pct: float,
    max_latency_regression_pct: float,
    allow_environment_mismatch: bool,
) -> dict[str, Any]:
    errors: list[str] = []

    def index(records: list[dict[str, Any]], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
        indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in records:
            if record.get("record_type") != "summary":
                continue
            key = identity(record)
            missing_fields = [field for field, value in zip(IDENTITY_FIELDS, key) if value is None]
            if missing_fields:
                errors.append(f"{label}: missing workload identity fields {missing_fields} in record")
            if key in indexed:
                errors.append(f"{label}: duplicate workload identity {identity_dict(key)}")
            indexed[key] = record
            if record.get("status") != "ok":
                errors.append(f"{label}: workload {identity_dict(key)} has status={record.get('status')!r}")
        return indexed

    baseline = index(baseline_records, "baseline")
    candidate = index(candidate_records, "candidate")
    if not baseline:
        errors.append("baseline has no summary records")
    if not candidate:
        errors.append("candidate has no summary records")
    if set(baseline) != set(candidate):
        missing = [identity_dict(k) for k in sorted(set(baseline) - set(candidate), key=str)]
        extra = [identity_dict(k) for k in sorted(set(candidate) - set(baseline), key=str)]
        if missing:
            errors.append(f"candidate is missing workload identities: {missing}")
        if extra:
            errors.append(f"candidate has unmatched workload identities: {extra}")

    workloads = []
    for key in sorted(set(baseline) & set(candidate), key=str):
        b_record, c_record = baseline[key], candidate[key]
        b_env, c_env = environment(b_record), environment(c_record)
        env_diffs = {field: {"baseline": b_env[field], "candidate": c_env[field]}
                     for field in ENVIRONMENT_FIELDS if b_env[field] != c_env[field]}
        strict_environment_match = not env_diffs
        metrics_b, metrics_c = median_metrics(b_record), median_metrics(c_record)
        failures = []
        for label, metrics in (("baseline", metrics_b), ("candidate", metrics_c)):
            for measurement_error in metrics["measurement_errors"]:
                failures.append(f"{label}: invalid measurement for workload {identity_dict(key)}: {measurement_error}")
        if metrics_b["run_count"] < min_runs or metrics_c["run_count"] < min_runs:
            failures.append(f"measured runs below minimum {min_runs}: baseline={metrics_b['run_count']}, candidate={metrics_c['run_count']}")
        if env_diffs and not allow_environment_mismatch:
            failures.append(f"environment mismatch: {env_diffs}")
        if metrics_b["run_count"] >= min_runs and metrics_c["run_count"] >= min_runs and not metrics_b["measurement_errors"] and not metrics_c["measurement_errors"]:
            try:
                throughput_change = percent_change(metrics_c["median_output_tokens_per_second"], metrics_b["median_output_tokens_per_second"])
                latency_change = percent_change(metrics_c["median_seconds"], metrics_b["median_seconds"])
            except ValueError as exc:
                failures.append(f"invalid baseline/candidate median for workload {identity_dict(key)}: {exc}")
                throughput_change = None
                latency_change = None
            if throughput_change is not None and throughput_change < -max_throughput_regression_pct:
                failures.append(f"throughput regression {abs(throughput_change):.3f}% exceeds {max_throughput_regression_pct:.3f}%")
            if latency_change is not None and latency_change > max_latency_regression_pct:
                failures.append(f"latency regression {latency_change:.3f}% exceeds {max_latency_regression_pct:.3f}%")
        else:
            throughput_change = None
            latency_change = None
        workload = {
            "identity": identity_dict(key),
            "baseline": {**metrics_b, "environment": b_env},
            "candidate": {**metrics_c, "environment": c_env},
            "throughput_change_pct": throughput_change,
            "latency_change_pct": latency_change,
            "environment_match": strict_environment_match,
            "strictly_comparable": strict_environment_match and not allow_environment_mismatch,
            "status": "ok" if not failures else "failed",
            "failures": failures,
        }
        workloads.append(workload)
        if failures:
            errors.extend([f"{identity_dict(key)}: {failure}" for failure in failures])

    return {
        "status": "passed" if not errors and workloads and all(w["status"] == "ok" for w in workloads) else "failed",
        "identity_fields": list(IDENTITY_FIELDS),
        "environment_fields": list(ENVIRONMENT_FIELDS),
        "min_runs": min_runs,
        "thresholds": {
            "max_throughput_regression_pct": max_throughput_regression_pct,
            "max_latency_regression_pct": max_latency_regression_pct,
        },
        "allow_environment_mismatch": allow_environment_mismatch,
        "strict_comparison": not allow_environment_mismatch and not any(w.get("environment_match") is False for w in workloads),
        "errors": errors,
        "workloads": workloads,
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def markdown(result: dict[str, Any], baseline_path: Path, candidate_path: Path) -> str:
    lines = [
        "# Offline Benchmark Regression Gate", "",
        f"Status: **{result['status']}**", "",
        f"Baseline: `{baseline_path}`  ", f"Candidate: `{candidate_path}`", "",
        f"Minimum measured runs: `{result['min_runs']}`; thresholds: throughput `-{result['thresholds']['max_throughput_regression_pct']}%`, latency `+{result['thresholds']['max_latency_regression_pct']}%`.",
        f"Strict environment comparison: `{result['strict_comparison']}`; allow mismatch: `{result['allow_environment_mismatch']}`.", "",
        "| Workload identity | Baseline median tok/s | Candidate median tok/s | Throughput change | Baseline median s | Candidate median s | Latency change | Environment | Status |", 
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for workload in result["workloads"]:
        identity_text = "; ".join(f"{k}={v}" for k, v in workload["identity"].items())
        b, c = workload["baseline"], workload["candidate"]
        fmt = lambda value: "-" if value is None else f"{value:.3f}"
        change = lambda value: "-" if value is None else f"{value:+.3f}%"
        lines.append(f"| `{identity_text}` | {fmt(b['median_output_tokens_per_second'])} | {fmt(c['median_output_tokens_per_second'])} | {change(workload['throughput_change_pct'])} | {fmt(b['median_seconds'])} | {fmt(c['median_seconds'])} | {change(workload['latency_change_pct'])} | `{workload['environment_match']}` | **{workload['status']}** |")
    if result["errors"]:
        lines += ["", "## Failures", ""]
        lines.extend(f"- {error}" for error in result["errors"])
    lines += ["", "## Identity fields", "", ", ".join(f"`{field}`" for field in result["identity_fields"]), "", "## Environment fields", "", ", ".join(f"`{field}`" for field in result["environment_fields"]), ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare existing benchmark JSONL summaries without running a benchmark")
    parser.add_argument("baseline_jsonl", type=Path)
    parser.add_argument("candidate_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--max-throughput-regression-pct", type=nonnegative_finite, default=10.0)
    parser.add_argument("--max-latency-regression-pct", type=nonnegative_finite, default=15.0)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.min_runs < 1:
        parser.error("--min-runs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline_records = read_jsonl_records(args.baseline_jsonl)
        candidate_records = read_jsonl_records(args.candidate_jsonl)
        result = compare_records(
            baseline_records, candidate_records, args.min_runs,
            args.max_throughput_regression_pct, args.max_latency_regression_pct,
            args.allow_environment_mismatch,
        )
    except Exception as exc:
        result = {
            "status": "failed", "identity_fields": list(IDENTITY_FIELDS),
            "environment_fields": list(ENVIRONMENT_FIELDS), "errors": [f"input error: {type(exc).__name__}: {exc}"],
            "workloads": [], "min_runs": args.min_runs,
            "thresholds": {"max_throughput_regression_pct": args.max_throughput_regression_pct, "max_latency_regression_pct": args.max_latency_regression_pct},
            "allow_environment_mismatch": args.allow_environment_mismatch, "strict_comparison": False,
        }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_json or args.output_dir / f"benchmark_regression_{timestamp}.json"
    md_path = args.output_md or args.output_dir / f"benchmark_regression_{timestamp}.md"
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    if not md_path.is_absolute():
        md_path = Path.cwd() / md_path
    result["baseline_jsonl"] = str(args.baseline_jsonl)
    result["candidate_jsonl"] = str(args.candidate_jsonl)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write(json_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write(md_path, markdown(result, args.baseline_jsonl, args.candidate_jsonl))
    print(json.dumps({"status": result["status"], "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
    for error in result.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
