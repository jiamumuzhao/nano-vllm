"""Run a controlled offline generation workload matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_schema import REPO_ROOT, environment_info, now_utc, write_jsonl_record


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3x3 offline workload matrix with strict benchmark settings")
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--engines", default="nano_eager,nano_graph,hf,vllm")
    parser.add_argument("--num-seqs-list", default="1,8,16")
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--preflight-output-len", type=int, default=4)
    parser.add_argument("--preflight-runs", type=int, default=1)
    parser.add_argument("--preflight-warmup-runs", type=int, default=0)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--matrix-dir", default="docs/benchmarks/matrix")
    parser.add_argument("--output-jsonl", default="docs/benchmarks/offline_matrix_2026-07-12.jsonl")
    parser.add_argument("--output-md", default="docs/benchmarks/offline_matrix_2026-07-12.md")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def workload_id(input_len: int, num_seqs: int) -> str:
    return f"input{input_len}_seqs{num_seqs}"


def benchmark_command(args: argparse.Namespace, *, input_len: int, num_seqs: int, output_len: int, warmup_runs: int, runs: int, output_jsonl: Path, output_md: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_offline.py"),
        "--engines", args.engines,
        "--model", args.model,
        "--num-seqs", str(num_seqs),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--warmup-runs", str(warmup_runs),
        "--runs", str(runs),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--output-jsonl", str(output_jsonl.relative_to(REPO_ROOT)),
        "--output-md", str(output_md.relative_to(REPO_ROOT)),
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_case(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def append_records(matrix_jsonl: Path, *, phase: str, input_len: int, num_seqs: int, source_jsonl: Path, source_md: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        item = dict(record)
        item["matrix_phase"] = phase
        item["matrix_workload"] = {"input_len": input_len, "num_seqs": num_seqs}
        item["matrix_source_jsonl"] = str(source_jsonl.relative_to(REPO_ROOT))
        item["matrix_source_md"] = str(source_md.relative_to(REPO_ROOT))
        write_jsonl_record(matrix_jsonl, item)


def failure_record(args: argparse.Namespace, *, phase: str, input_len: int, num_seqs: int, command: list[str], completed: subprocess.CompletedProcess[str], source_jsonl: Path, source_md: Path) -> dict[str, Any]:
    return {
        "record_type": "matrix_failure",
        "status": "failed",
        "created_at": now_utc(),
        "matrix_phase": phase,
        "matrix_workload": {"input_len": input_len, "num_seqs": num_seqs},
        "matrix_source_jsonl": str(source_jsonl.relative_to(REPO_ROOT)),
        "matrix_source_md": str(source_md.relative_to(REPO_ROOT)),
        "command": " ".join(shlex.quote(part) for part in command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "environment": environment_info(),
        "workload": {
            "model": args.model,
            "engines": args.engines,
            "output_len": args.output_len,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "dtype": args.dtype,
            "seed": args.seed,
            "temperature": args.temperature,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    }


def summary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_type") == "summary"]


def find_summary(summaries: list[dict[str, Any]], engine: str) -> dict[str, Any] | None:
    for record in summaries:
        if record.get("engine") == engine:
            return record
    return None


def rate(record: dict[str, Any] | None) -> float | None:
    if not record or record.get("status") != "ok":
        return None
    return (record.get("summary") or {}).get("mean_output_tokens_per_second")


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_overview(path: Path, args: argparse.Namespace, formal_by_workload: dict[tuple[int, int], list[dict[str, Any]]], preflight_status: dict[tuple[int, int], str], matrix_jsonl: Path, command_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Offline Workload Matrix Benchmark - 2026-07-12\n\n")
        handle.write("This is a controlled offline generation workload matrix. It does not measure TTFT, TPOT, P95/P99 latency, dynamic request arrivals, streaming behavior, or service quality.\n\n")
        handle.write(f"Run timestamp: `{now_utc()}`\n\n")
        handle.write(f"Summary JSONL: [{matrix_jsonl.relative_to(path.parent).as_posix()}]({matrix_jsonl.relative_to(path.parent).as_posix()})\n\n")
        handle.write("## Command\n\n```bash\n")
        handle.write(command_text + "\n")
        handle.write("```\n\n")
        handle.write("## Environment\n\n")
        handle.write(f"- Python: `{sys.executable}`\n")
        handle.write(f"- Git revision: `{(env.get('git') or {}).get('commit')}`; dirty: `{(env.get('git') or {}).get('is_dirty')}`\n")
        handle.write(f"- PyTorch: `{env.get('torch')}`; CUDA runtime: `{env.get('cuda_runtime')}`; Triton: `{env.get('triton')}`; Transformers: `{env.get('transformers')}`\n")
        for gpu in env.get("gpus", []) or []:
            handle.write(f"- GPU {gpu.get('index')}: `{gpu.get('name')}`, capability `{gpu.get('capability')}`, memory `{gpu.get('total_memory_bytes')}` bytes\n")
        handle.write("\n## Unified Parameters\n\n")
        handle.write(f"- Model: `{args.model}`\n")
        handle.write(f"- Engines: `{args.engines}`\n")
        handle.write(f"- Matrix num-seqs: `{args.num_seqs_list}`; input-lens: `{args.input_lens}`\n")
        handle.write(f"- Output length: `{args.output_len}`; warmup runs: `{args.warmup_runs}`; measured runs: `{args.runs}`\n")
        handle.write(f"- Dtype: `{args.dtype}`; TP: `{args.tensor_parallel_size}`; temperature: `{args.temperature}`; seed: `{args.seed}`\n")
        handle.write(f"- Max model len: `{args.max_model_len}`; max batched tokens: `{args.max_num_batched_tokens}`; max num seqs equals workload `num_seqs`\n")
        handle.write("- Fixed-output, prompt-token, and measured-seed semantics are inherited from `scripts/benchmark_offline.py`.\n\n")
        handle.write("## Preflight Status\n\n")
        handle.write("| input_len | num_seqs | status |\n| ---: | ---: | --- |\n")
        for input_len, num_seqs in sorted(preflight_status):
            handle.write(f"| {input_len} | {num_seqs} | {preflight_status[(input_len, num_seqs)]} |\n")
        handle.write("\n## Workload Results\n\n")
        for input_len, num_seqs in sorted(formal_by_workload):
            summaries = summary_records(formal_by_workload[(input_len, num_seqs)])
            handle.write(f"### input_len={input_len}, num_seqs={num_seqs}\n\n")
            handle.write("| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |\n")
            handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
            for engine in engines:
                rec = find_summary(summaries, engine)
                if rec and rec.get("status") == "ok":
                    s = rec.get("summary") or {}
                    seconds = ", ".join(fmt(x, 4) for x in s.get("run_seconds", []))
                    source = rec.get("matrix_source_jsonl", "")
                    handle.write(f"| {engine} | ok | {fmt(s.get('mean_output_tokens_per_second'))} | {fmt(s.get('stddev_output_tokens_per_second'))} | {fmt(s.get('min_output_tokens_per_second'))} | {fmt(s.get('max_output_tokens_per_second'))} | {seconds} | `{source}` |\n")
                elif rec:
                    handle.write(f"| {engine} | failed | - | - | - | - | - | `{rec.get('matrix_source_jsonl', '')}` |\n")
                else:
                    handle.write(f"| {engine} | not run | - | - | - | - | - | - |\n")
            nano_graph = rate(find_summary(summaries, "nano_graph"))
            vllm = rate(find_summary(summaries, "vllm"))
            nano_eager = rate(find_summary(summaries, "nano_eager"))
            hf = rate(find_summary(summaries, "hf"))
            handle.write("\n")
            handle.write(f"- Nano Graph / vLLM: `{fmt(ratio(nano_graph, vllm), 3)}`\n")
            handle.write(f"- Nano eager / HF: `{fmt(ratio(nano_eager, hf), 3)}`\n\n")
        handle.write("## Conclusion Table\n\n")
        handle.write("| input_len | num_seqs | Nano Graph tok/s | vLLM tok/s | Faster | Nano Graph / vLLM | Nano eager / HF |\n")
        handle.write("| ---: | ---: | ---: | ---: | --- | ---: | ---: |\n")
        for input_len, num_seqs in sorted(formal_by_workload):
            summaries = summary_records(formal_by_workload[(input_len, num_seqs)])
            ng = rate(find_summary(summaries, "nano_graph"))
            vv = rate(find_summary(summaries, "vllm"))
            ne = rate(find_summary(summaries, "nano_eager"))
            hh = rate(find_summary(summaries, "hf"))
            if ng is None or vv is None:
                faster = "incomplete"
            elif ng > vv:
                faster = "Nano Graph"
            elif vv > ng:
                faster = "vLLM"
            else:
                faster = "tie"
            handle.write(f"| {input_len} | {num_seqs} | {fmt(ng)} | {fmt(vv)} | {faster} | {fmt(ratio(ng, vv), 3)} | {fmt(ratio(ne, hh), 3)} |\n")
        handle.write("\n## Boundary\n\n")
        handle.write("These results apply to Qwen3-0.6B on this 2 x RTX 2080 Ti host under the listed offline workloads. They should not be generalized to production serving behavior, other models, other GPUs, online latency, or dynamic traffic.\n")


def main() -> None:
    args = parse_args()
    input_lens = parse_csv_ints(args.input_lens)
    num_seqs_list = parse_csv_ints(args.num_seqs_list)
    matrix_dir = (REPO_ROOT / args.matrix_dir).resolve()
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_jsonl = (REPO_ROOT / args.output_jsonl).resolve()
    matrix_md = (REPO_ROOT / args.output_md).resolve()
    if matrix_jsonl.exists():
        matrix_jsonl.unlink()
    formal_by_workload: dict[tuple[int, int], list[dict[str, Any]]] = {}
    preflight_status: dict[tuple[int, int], str] = {}
    command_text = " ".join(shlex.quote(part) for part in [sys.executable, "scripts/benchmark_offline_matrix.py"] + sys.argv[1:])
    for input_len in input_lens:
        for num_seqs in num_seqs_list:
            wid = workload_id(input_len, num_seqs)
            if not args.skip_preflight:
                pre_jsonl = matrix_dir / f"preflight_{wid}.jsonl"
                pre_md = matrix_dir / f"preflight_{wid}.md"
                pre_cmd = benchmark_command(args, input_len=input_len, num_seqs=num_seqs, output_len=args.preflight_output_len, warmup_runs=args.preflight_warmup_runs, runs=args.preflight_runs, output_jsonl=pre_jsonl, output_md=pre_md)
                print(f"PREFLIGHT {wid}", flush=True)
                completed = run_case(pre_cmd)
                pre_records = load_jsonl(pre_jsonl)
                if completed.returncode != 0:
                    rec = failure_record(args, phase="preflight", input_len=input_len, num_seqs=num_seqs, command=pre_cmd, completed=completed, source_jsonl=pre_jsonl, source_md=pre_md)
                    write_jsonl_record(matrix_jsonl, rec)
                    preflight_status[(input_len, num_seqs)] = "failed-driver"
                    continue
                append_records(matrix_jsonl, phase="preflight", input_len=input_len, num_seqs=num_seqs, source_jsonl=pre_jsonl, source_md=pre_md, records=pre_records)
                summaries = summary_records(pre_records)
                if any(record.get("status") != "ok" for record in summaries) or len(summaries) != len([e for e in args.engines.split(',') if e.strip()]):
                    preflight_status[(input_len, num_seqs)] = "failed-engine"
                    continue
                preflight_status[(input_len, num_seqs)] = "ok"
            else:
                preflight_status[(input_len, num_seqs)] = "skipped"
            formal_jsonl = matrix_dir / f"formal_{wid}.jsonl"
            formal_md = matrix_dir / f"formal_{wid}.md"
            formal_cmd = benchmark_command(args, input_len=input_len, num_seqs=num_seqs, output_len=args.output_len, warmup_runs=args.warmup_runs, runs=args.runs, output_jsonl=formal_jsonl, output_md=formal_md)
            print(f"FORMAL {wid}", flush=True)
            completed = run_case(formal_cmd)
            formal_records = load_jsonl(formal_jsonl)
            if completed.returncode != 0:
                rec = failure_record(args, phase="formal", input_len=input_len, num_seqs=num_seqs, command=formal_cmd, completed=completed, source_jsonl=formal_jsonl, source_md=formal_md)
                write_jsonl_record(matrix_jsonl, rec)
                formal_by_workload[(input_len, num_seqs)] = [rec]
                continue
            append_records(matrix_jsonl, phase="formal", input_len=input_len, num_seqs=num_seqs, source_jsonl=formal_jsonl, source_md=formal_md, records=formal_records)
            formal_by_workload[(input_len, num_seqs)] = [dict(record, matrix_source_jsonl=str(formal_jsonl.relative_to(REPO_ROOT)), matrix_source_md=str(formal_md.relative_to(REPO_ROOT))) for record in formal_records]
    write_overview(matrix_md, args, formal_by_workload, preflight_status, matrix_jsonl, command_text)
    print(f"Wrote {matrix_jsonl}")
    print(f"Wrote {matrix_md}")


if __name__ == "__main__":
    main()
