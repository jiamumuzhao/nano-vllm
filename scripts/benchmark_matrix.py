"""Run a small generate-throughput matrix and write JSONL/Markdown results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_result(stdout: str) -> dict:
    lines = stdout.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "{":
            start = idx
            break
    if start is None:
        raise ValueError(f"benchmark output did not contain JSON:\n{stdout}")
    return json.loads("\n".join(lines[start:]))


def run_case(args, batch_size: int, input_len: int, mode: str) -> dict:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_generate.py"),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--num-seqs",
        str(batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--input-len",
        str(input_len),
        "--output-len",
        str(args.output_len),
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs",
        str(args.runs),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if mode == "eager":
        command.append("--enforce-eager")
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    result = parse_result(completed.stdout)
    return {
        "mode": mode,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": args.output_len,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "mean_end_to_end_output_tokens_per_second": result["mean_end_to_end_output_tokens_per_second"],
        "run_seconds": [item["seconds"] for item in result["measurements"]],
        "initialization_seconds": result["initialization_seconds"],
        "cuda_device_names": result["cuda_device_names"],
    }


def write_markdown(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda item: (item["input_len"], item["batch_size"], item["mode"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Performance Matrix\n\n")
        handle.write("End-to-end output tokens/s around full `llm.generate()` after warmup.\n\n")
        handle.write("| input_len | batch | eager tok/s | graph tok/s | graph/eager |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        grouped = {}
        for item in rows:
            grouped.setdefault((item["input_len"], item["batch_size"]), {})[item["mode"]] = item
        for input_len, batch_size in sorted(grouped):
            eager = grouped[(input_len, batch_size)].get("eager")
            graph = grouped[(input_len, batch_size)].get("graph")
            eager_rate = eager["mean_end_to_end_output_tokens_per_second"] if eager else None
            graph_rate = graph["mean_end_to_end_output_tokens_per_second"] if graph else None
            ratio = graph_rate / eager_rate if eager_rate and graph_rate else None
            eager_text = f"{eager_rate:.2f}" if eager_rate is not None else "-"
            graph_text = f"{graph_rate:.2f}" if graph_rate is not None else "-"
            ratio_text = f"{ratio:.2f}x" if ratio is not None else "-"
            handle.write(
                f"| {input_len} | {batch_size} | "
                f"{eager_text} | {graph_text} | {ratio_text} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8,12,16")
    parser.add_argument("--input-lens", default="128,512,1024")
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-jsonl", default="docs/perf_matrix.jsonl")
    parser.add_argument("--output-md", default="docs/perf_matrix.md")
    args = parser.parse_args()

    batch_sizes = parse_csv_ints(args.batch_sizes)
    input_lens = parse_csv_ints(args.input_lens)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    results = []
    jsonl_path = REPO_ROOT / args.output_jsonl
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for input_len in input_lens:
            for batch_size in batch_sizes:
                for mode in modes:
                    print(f"RUN input_len={input_len} batch={batch_size} mode={mode}", flush=True)
                    result = run_case(args, batch_size, input_len, mode)
                    results.append(result)
                    jsonl.write(json.dumps(result, sort_keys=True) + "\n")
                    jsonl.flush()
                    rate = result["mean_end_to_end_output_tokens_per_second"]
                    print(f"DONE input_len={input_len} batch={batch_size} mode={mode} tok/s={rate:.2f}", flush=True)
    write_markdown(REPO_ROOT / args.output_md, results)


if __name__ == "__main__":
    main()
