"""Profile Nano-vLLM offline generation for long-context analysis."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_schema import deterministic_prompt_ids, environment_info, now_utc, set_seed, synchronize_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Nano-vLLM prefill/decode behavior for one offline workload")
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output-dir", default="docs/profiles")
    parser.add_argument("--profile-steps", type=int, default=12, help="Number of generation steps captured by torch.profiler")
    parser.add_argument("--top-k", type=int, default=20, help="Number of profiler events to keep in JSON/Markdown")
    parser.add_argument("--save-trace", action="store_true", help="Save a Chrome trace in the output directory")
    parser.add_argument(
        "--warmup-prompt-mode",
        choices=("same", "different"),
        default="same",
        help="Use the measured prompt during warmup to reproduce the strict benchmark's prefix-cache behavior, or use a different prompt to profile uncached prefill.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_seconds(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def profiler_event_rows(prof: Any, sort_by: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in prof.key_averages():
        cuda_us = float(getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0)) or 0.0)
        cpu_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        count = int(getattr(event, "count", 0) or 0)
        rows.append({
            "name": getattr(event, "key", ""),
            "count": count,
            "cpu_time_total_us": cpu_us,
            "cuda_time_total_us": cuda_us,
            "cpu_time_avg_us": cpu_us / count if count else None,
            "cuda_time_avg_us": cuda_us / count if count else None,
        })
    key = "cuda_time_total_us" if sort_by == "cuda" else "cpu_time_total_us"
    rows.sort(key=lambda item: item[key], reverse=True)
    return rows[:limit]


def make_markdown(result: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines: list[str] = []
    workload = result["workload"]
    metrics = result["metrics"]
    lines.append(f"# Nano-vLLM Profile - {workload['mode']} input{workload['input_len']} seqs{workload['num_seqs']}")
    lines.append("")
    lines.append(f"- Created at: `{result['created_at']}`")
    lines.append(f"- Command: `{result['command']}`")
    lines.append(f"- Prompt SHA-256: `{result['prompt']['prompt_token_sha256']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Initialization seconds | {fmt(metrics['initialization_seconds'])} |")
    lines.append(f"| Prefill tokens | {metrics['prefill']['tokens']} |")
    lines.append(f"| Prefill synchronized seconds | {fmt(metrics['prefill']['synchronized_wall_seconds'])} |")
    lines.append(f"| Prefill tok/s | {fmt(metrics['prefill']['tokens_per_second'], 2)} |")
    lines.append(f"| Decode tokens | {metrics['decode']['tokens']} |")
    lines.append(f"| Decode synchronized seconds | {fmt(metrics['decode']['synchronized_wall_seconds'])} |")
    lines.append(f"| Decode tok/s | {fmt(metrics['decode']['tokens_per_second'], 2)} |")
    lines.append(f"| Decode step mean seconds | {fmt(metrics['decode_step_synchronized_seconds']['mean'])} |")
    lines.append(f"| Decode step median seconds | {fmt(metrics['decode_step_synchronized_seconds']['median'])} |")
    lines.append(f"| Decode step p95 seconds | {fmt(metrics['decode_step_synchronized_seconds']['p95'])} |")
    lines.append(f"| Max CUDA allocated bytes | {metrics['cuda_memory']['max_allocated_bytes']} |")
    lines.append(f"| Max CUDA reserved bytes | {metrics['cuda_memory']['max_reserved_bytes']} |")
    lines.append("")
    lines.append("## Top CUDA Events")
    lines.append("")
    lines.append("| Event | Count | CUDA total ms | CPU total ms |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in result["profiler"]["top_cuda_events"]:
        lines.append(f"| `{row['name']}` | {row['count']} | {row['cuda_time_total_us'] / 1000.0:.3f} | {row['cpu_time_total_us'] / 1000.0:.3f} |")
    lines.append("")
    lines.append("## Top CPU Events")
    lines.append("")
    lines.append("| Event | Count | CPU total ms | CUDA total ms |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in result["profiler"]["top_cpu_events"]:
        lines.append(f"| `{row['name']}` | {row['count']} | {row['cpu_time_total_us'] / 1000.0:.3f} | {row['cuda_time_total_us'] / 1000.0:.3f} |")
    lines.append("")
    return "\n".join(lines)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nanovllm import LLM, SamplingParams

    prompt_info = deterministic_prompt_ids(args.model, args.num_seqs, args.input_len, args.seed)
    warmup_prompt_info = (
        prompt_info
        if args.warmup_prompt_mode == "same"
        else deterministic_prompt_ids(args.model, args.num_seqs, args.input_len, args.seed + 1_000_000)
    )
    prompts = prompt_info["prompt_token_ids"]
    warmup_prompts = warmup_prompt_info["prompt_token_ids"]
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)
        for _ in prompts
    ]

    started = time.perf_counter()
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.mode == "eager",
        dtype=args.dtype,
    )
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - started

    try:
        for warmup_idx in range(args.warmup_runs):
            set_seed(args.seed + warmup_idx)
            llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
            synchronize_cuda()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for prompt, params in zip(prompts, sampling_params):
            llm.add_request(prompt, params)

        step_records: list[dict[str, Any]] = []
        outputs: dict[int, list[int]] = {}
        profiled_steps = 0
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        trace_path = None
        with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=False, acc_events=True) as prof:
            step_idx = 0
            while not llm.is_finished():
                synchronize_cuda()
                t0 = time.perf_counter()
                output, num_tokens, token_events = llm.step_with_events()
                t1 = time.perf_counter()
                synchronize_cuda()
                t2 = time.perf_counter()
                phase = "prefill" if num_tokens > 0 else "decode"
                generated = len(token_events) if phase == "decode" else 0
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                step_records.append({
                    "step_index": step_idx,
                    "phase": phase,
                    "scheduled_tokens": int(num_tokens) if phase == "prefill" else int(-num_tokens),
                    "generated_tokens": generated,
                    "cpu_wall_seconds": t1 - t0,
                    "synchronized_wall_seconds": t2 - t0,
                })
                if profiled_steps < args.profile_steps:
                    prof.step()
                    profiled_steps += 1
                step_idx += 1
        if args.save_trace:
            trace_path = Path(args.output_dir) / f"nano_profile_{args.mode}_input{args.input_len}_seqs{args.num_seqs}.trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path))

        output_token_count = sum(len(tokens) for tokens in outputs.values())
        expected_output_tokens = args.num_seqs * args.output_len
        prefill_steps = [item for item in step_records if item["phase"] == "prefill"]
        decode_steps = [item for item in step_records if item["phase"] == "decode"]
        prefill_tokens = sum(item["scheduled_tokens"] for item in prefill_steps)
        decode_tokens = sum(item["generated_tokens"] for item in decode_steps)
        prefill_sync = sum(item["synchronized_wall_seconds"] for item in prefill_steps)
        decode_sync = sum(item["synchronized_wall_seconds"] for item in decode_steps)
        prefill_cpu = sum(item["cpu_wall_seconds"] for item in prefill_steps)
        decode_cpu = sum(item["cpu_wall_seconds"] for item in decode_steps)

        result = {
            "created_at": now_utc(),
            "command": " ".join([sys.executable, *sys.argv]),
            "environment": environment_info(),
            "workload": {
                "model": args.model,
                "mode": args.mode,
                "num_seqs": args.num_seqs,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "warmup_runs": args.warmup_runs,
                "dtype": args.dtype,
                "seed": args.seed,
                "measured_seed": args.seed + 10_000,
                "temperature": args.temperature,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "profile_steps": args.profile_steps,
            },
            "prompt": {
                "input_token_count_per_sequence": args.input_len,
                "num_sequences": args.num_seqs,
                "prompt_token_sha256": prompt_info["prompt_token_sha256"],
                "tokenizer_class": prompt_info["tokenizer_class"],
                "vocab_size": prompt_info["vocab_size"],
            },
            "sampling": {
                "temperature": args.temperature,
                "fixed_output_length": True,
                "ignore_eos": True,
                "is_greedy": False,
                "warmup_seeds": [args.seed + i for i in range(args.warmup_runs)],
                "measured_seed": args.seed + 10_000,
                "warmup_prompt_token_sha256": warmup_prompt_info["prompt_token_sha256"],
                "measured_prompt_differs_from_warmup": warmup_prompt_info["prompt_token_sha256"] != prompt_info["prompt_token_sha256"],
                "warmup_prompt_mode": args.warmup_prompt_mode,
            },
            "metrics": {
                "initialization_seconds": initialization_seconds,
                "output_tokens": output_token_count,
                "expected_output_tokens": expected_output_tokens,
                "all_outputs_fixed_length": output_token_count == expected_output_tokens,
                "prefill": {
                    "steps": len(prefill_steps),
                    "tokens": prefill_tokens,
                    "cpu_wall_seconds": prefill_cpu,
                    "synchronized_wall_seconds": prefill_sync,
                    "tokens_per_second": prefill_tokens / prefill_sync if prefill_sync else None,
                },
                "decode": {
                    "steps": len(decode_steps),
                    "tokens": decode_tokens,
                    "cpu_wall_seconds": decode_cpu,
                    "synchronized_wall_seconds": decode_sync,
                    "tokens_per_second": decode_tokens / decode_sync if decode_sync else None,
                },
                "decode_step_cpu_seconds": summarize_seconds([item["cpu_wall_seconds"] for item in decode_steps]),
                "decode_step_synchronized_seconds": summarize_seconds([item["synchronized_wall_seconds"] for item in decode_steps]),
                "cuda_memory": {
                    "max_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    "max_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                },
            },
            "steps": step_records,
            "profiler": {
                "profiled_steps": profiled_steps,
                "top_cuda_events": profiler_event_rows(prof, "cuda", args.top_k),
                "top_cpu_events": profiler_event_rows(prof, "cpu", args.top_k),
                "chrome_trace_path": str(trace_path) if trace_path else None,
            },
        }
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_profile(args)
    stem = f"nano_profile_{args.mode}_input{args.input_len}_seqs{args.num_seqs}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_path.write_text(make_markdown(result) + "\n")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
