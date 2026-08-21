"""Run reproducible offline generation benchmarks and write JSONL/Markdown."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_schema import REPO_ROOT, compact_error, environment_info, now_utc, write_jsonl_record

RESULT_MARKER = "###BENCHMARK_RESULT_JSON###"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified offline generation benchmark runner")
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--engines", default="nano_eager,nano_graph,hf", help="Comma-separated: nano_eager,nano_graph,hf,vllm")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output-jsonl", default="docs/benchmarks/offline_unified.jsonl")
    parser.add_argument("--output-md", default="docs/benchmarks/offline_unified.md")
    return parser.parse_args()


def worker_command(args: argparse.Namespace, engine: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_offline_worker.py"),
        "--engine", engine,
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--warmup-runs", str(args.warmup_runs),
        "--runs", str(args.runs),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]


def parse_worker_result(stdout: str) -> dict[str, Any]:
    if RESULT_MARKER not in stdout:
        raise ValueError("worker output did not contain result marker")
    payload = stdout.split(RESULT_MARKER, 1)[1].strip().splitlines()[0]
    return json.loads(payload)


def run_engine(args: argparse.Namespace, engine: str, jsonl_path: Path) -> dict[str, Any]:
    command = worker_command(args, engine)
    started_at = now_utc()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    command_text = " ".join(shlex.quote(part) for part in command)
    if completed.returncode != 0:
        record = {
            "record_type": "summary",
            "status": "failed",
            "created_at": now_utc(),
            "started_at": started_at,
            "engine": engine,
            "mode": engine.replace("nano_", "") if engine.startswith("nano_") else engine,
            "model": args.model,
            "dtype": args.dtype,
            "seed": args.seed,
            "temperature": args.temperature,
            "workload": vars(args) | {"engine": engine},
            "command": command_text,
            "failure_stage": "worker_process",
            "error_summary": compact_error(completed.stderr, completed.stdout),
            "environment": environment_info(),
        }
        write_jsonl_record(jsonl_path, record)
        return record
    try:
        summary = parse_worker_result(completed.stdout)
    except Exception as exc:
        record = {
            "record_type": "summary",
            "status": "failed",
            "created_at": now_utc(),
            "started_at": started_at,
            "engine": engine,
            "mode": engine.replace("nano_", "") if engine.startswith("nano_") else engine,
            "model": args.model,
            "dtype": args.dtype,
            "seed": args.seed,
            "temperature": args.temperature,
            "workload": vars(args) | {"engine": engine},
            "command": command_text,
            "failure_stage": "result_parse",
            "error_summary": f"{exc}\n{compact_error(completed.stderr, completed.stdout)}",
            "environment": environment_info(),
        }
        write_jsonl_record(jsonl_path, record)
        return record
    summary["command"] = command_text
    for measurement in summary.get("measurements", []):
        run_record = {
            key: value for key, value in summary.items()
            if key not in {"measurements", "summary", "record_type"}
        }
        run_record["record_type"] = "run"
        run_record["run"] = measurement
        run_record["run_index"] = measurement.get("run_index")
        run_record["seed"] = measurement.get("seed")
        write_jsonl_record(jsonl_path, run_record)
    write_jsonl_record(jsonl_path, summary)
    return summary


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, args: argparse.Namespace, records: list[dict[str, Any]], jsonl_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = next((record.get("environment") for record in records if record.get("environment")), environment_info())
    command = " ".join(shlex.quote(part) for part in [sys.executable, "scripts/benchmark_offline.py"] + sys.argv[1:])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Unified Offline Generation Benchmark\n\n")
        handle.write(f"Run timestamp: `{now_utc()}`\n\n")
        jsonl_link = jsonl_path.relative_to(path.parent).as_posix()
        handle.write(f"Raw JSONL: [`{jsonl_path.as_posix()}`]({jsonl_link})\n\n")
        handle.write("## Command\n\n")
        handle.write("```bash\n")
        handle.write(command + "\n")
        handle.write("```\n\n")
        handle.write("## Workload\n\n")
        handle.write(f"- Model: `{args.model}`\n")
        handle.write(f"- Engines: `{args.engines}`\n")
        handle.write(f"- Sequences: `{args.num_seqs}`\n")
        handle.write(f"- Input tokens per sequence: `{args.input_len}`\n")
        handle.write(f"- Output tokens per sequence: `{args.output_len}`\n")
        handle.write(f"- Warmup runs: `{args.warmup_runs}`\n")
        handle.write(f"- Measured runs: `{args.runs}`\n")
        handle.write(f"- Dtype: `{args.dtype}`\n")
        handle.write(f"- Base seed: `{args.seed}`\n")
        handle.write(f"- Measured run seeds: `{[args.seed + 10000 + i for i in range(args.runs)]}`\n")
        handle.write(f"- Temperature: `{args.temperature}`; stochastic sampling, not greedy\n")
        handle.write(f"- Max model length: `{args.max_model_len}`; max sequences: `{args.max_num_seqs}`; max batched tokens: `{args.max_num_batched_tokens}`\n")
        handle.write(f"- Fixed output length: Nano-vLLM uses `ignore_eos=True`; Hugging Face uses `min_new_tokens=max_new_tokens` and `eos_token_id=None`; vLLM uses `ignore_eos=True` plus `min_tokens=max_tokens`.\n\n")
        handle.write("## Environment\n\n")
        handle.write(f"- Python: `{env.get('python_executable')}` / `{str(env.get('python', '')).split()[0]}`\n")
        handle.write(f"- PyTorch: `{env.get('torch')}`\n")
        handle.write(f"- CUDA runtime: `{env.get('cuda_runtime')}`\n")
        handle.write(f"- Triton: `{env.get('triton')}`\n")
        handle.write(f"- Transformers: `{env.get('transformers')}`\n")
        git = env.get("git", {}) or {}
        handle.write(f"- Git revision: `{git.get('commit')}`; dirty: `{git.get('is_dirty')}`\n")
        for gpu in env.get("gpus", []) or []:
            handle.write(f"- GPU {gpu.get('index')}: `{gpu.get('name')}`, capability `{gpu.get('capability')}`, memory `{gpu.get('total_memory_bytes')}` bytes\n")
        handle.write("\n## Results\n\n")
        handle.write("| Engine | Mode | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Seeds | JSONL |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for record in records:
            if record.get("record_type") != "summary":
                continue
            status = record.get("status")
            summary = record.get("summary") or {}
            seconds = summary.get("run_seconds") or []
            seeds = [item.get("seed") for item in record.get("measurements", [])]
            if status == "ok":
                handle.write(
                    f"| {record.get('engine')} | {record.get('mode')} | {status} | "
                    f"{fmt(summary.get('mean_output_tokens_per_second'))} | "
                    f"{fmt(summary.get('stddev_output_tokens_per_second'))} | "
                    f"{fmt(summary.get('min_output_tokens_per_second'))} | "
                    f"{fmt(summary.get('max_output_tokens_per_second'))} | "
                    f"{', '.join(fmt(x, 4) for x in seconds)} | "
                    f"{', '.join(str(x) for x in seeds)} | `{jsonl_path.relative_to(REPO_ROOT).as_posix()}` |\n"
                )
            else:
                handle.write(f"| {record.get('engine')} | {record.get('mode')} | failed | - | - | - | - | - | - | `{jsonl_path.relative_to(REPO_ROOT).as_posix()}` |\n")
        failures = [record for record in records if record.get("status") != "ok"]
        if failures:
            handle.write("\n## Failures\n\n")
            for record in failures:
                handle.write(f"### {record.get('engine')}\n\n")
                handle.write(f"- Stage: `{record.get('failure_stage')}`\n")
                handle.write("```text\n")
                handle.write(str(record.get("error_summary", "")).strip() + "\n")
                handle.write("```\n\n")
        handle.write("## Fairness notes\n\n")
        handle.write("- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.\n")
        handle.write("- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.\n")
        handle.write("- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.\n")
        handle.write("- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.\n")
        handle.write("- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.\n")


def main() -> None:
    args = parse_args()
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    valid = {"nano_eager", "nano_graph", "hf", "vllm"}
    unknown = sorted(set(engines) - valid)
    if unknown:
        raise SystemExit(f"unknown engines: {unknown}")
    jsonl_path = (REPO_ROOT / args.output_jsonl).resolve()
    md_path = (REPO_ROOT / args.output_md).resolve()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()
    records = []
    for engine in engines:
        print(f"RUN {engine}", flush=True)
        record = run_engine(args, engine, jsonl_path)
        records.append(record)
        if record.get("status") == "ok":
            rate = record.get("summary", {}).get("mean_output_tokens_per_second")
            print(f"DONE {engine} mean_tok_s={rate:.2f}", flush=True)
        else:
            print(f"FAILED {engine}: {record.get('failure_stage')}", flush=True)
    write_markdown(md_path, args, records, jsonl_path)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
