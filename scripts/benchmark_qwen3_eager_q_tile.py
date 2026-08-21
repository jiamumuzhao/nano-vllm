"""End-to-end eager Qwen3-0.6B serial vs parallel paged-prefill benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def _ints(value):
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _summary(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(torch.tensor(values).quantile(0.50).item()),
        "p95_ms": float(torch.tensor(values).quantile(0.95).item()),
    }


def _run_once(engine, prompt_tokens, output_len, seed):
    torch.manual_seed(seed)
    params = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    for prompt in prompt_tokens:
        engine.add_request(prompt, params)
    start_time = time.perf_counter()
    prefill_ms = 0.0
    prefill_tokens = 0
    decode_latencies = []
    decode_tokens = 0
    first_token_time = None
    while not engine.is_finished():
        step_start = time.perf_counter()
        _, num_tokens, token_events = engine.step_with_events()
        step_ms = (time.perf_counter() - step_start) * 1000.0
        if num_tokens > 0:
            # num_tokens is Scheduler's sum(seq.num_scheduled_tokens), so this
            # also remains correct when chunked prefill takes multiple steps.
            prefill_ms += step_ms
            prefill_tokens += num_tokens
        else:
            decode_latencies.append(step_ms)
            decode_tokens += -num_tokens
        # Scheduler.postprocess emits token_events in the same step in which a
        # sequence completes prefill and its first sampled token is available.
        if first_token_time is None and token_events:
            first_token_time = time.perf_counter()
    end_to_end_ms = (time.perf_counter() - start_time) * 1000.0
    if prefill_tokens <= 0 or prefill_ms <= 0 or not decode_latencies or first_token_time is None:
        raise RuntimeError("benchmark run did not contain complete prefill/decode token events")
    decode_ms = statistics.fsum(decode_latencies)
    return {
        "ttft_ms": (first_token_time - start_time) * 1000.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tokens * 1000.0 / prefill_ms,
        "decode_tokens": decode_tokens,
        "decode_step_ms": statistics.fmean(decode_latencies),
        "decode_tok_s": decode_tokens * 1000.0 / decode_ms,
        "end_to_end_ms": end_to_end_ms,
        "end_to_end_tok_s": len(prompt_tokens) * output_len * 1000.0 / end_to_end_ms,
        "decode_steps": len(decode_latencies),
    }


def run_mode(engine, batch, input_len, output_len, parallel, warmups, repeats, seed):
    prompt_tokens = [[1] * input_len for _ in range(batch)]
    for warmup in range(warmups):
        _run_once(engine, prompt_tokens, output_len, seed + warmup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    runs = [_run_once(engine, prompt_tokens, output_len, seed + warmups + i)
            for i in range(repeats)]
    torch.cuda.synchronize()
    return {
        "batch_size": batch, "input_len": input_len, "output_len": output_len,
        "parallel": parallel, "status": "ok", "seed": seed,
        "warmup_runs": warmups, "repeat_runs": repeats,
        "ttft": _summary([r["ttft_ms"] for r in runs]),
        "prefill_latency": _summary([r["prefill_ms"] for r in runs]),
        "prefill_tok_s": _summary([r["prefill_tok_s"] for r in runs]),
        "decode_step_latency": _summary([r["decode_step_ms"] for r in runs]),
        "decode_tok_s": _summary([r["decode_tok_s"] for r in runs]),
        "end_to_end_tok_s": _summary([r["end_to_end_tok_s"] for r in runs]),
        "raw_runs": runs,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def markdown(data):
    grouped = {}
    for r in data["results"]:
        grouped.setdefault((r["input_len"], r["batch_size"]), {})[r["parallel"]] = r
    lines = ["# Qwen3-0.6B eager end-to-end Q-tile benchmark", "",
             "float16, tensor parallel size 1, CUDA Graph disabled.", "",
             "| Input | Batch | Path | TTFT ms | Prefill ms | Prefill tok/s | Decode step ms | Decode tok/s | E2E tok/s | Peak MB | Speedup note | Status |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for (input_len, batch), paths in sorted(grouped.items()):
        serial = paths.get(False)
        for parallel in (False, True):
            r = paths.get(parallel)
            if not r:
                continue
            peak = r["peak_memory_allocated_bytes"] / 1024**2
            note = "—"
            if parallel and serial:
                note = f"TTFT {serial['ttft']['mean_ms'] / r['ttft']['mean_ms']:.3f}x; E2E {r['end_to_end_tok_s']['mean_ms'] / serial['end_to_end_tok_s']['mean_ms']:.3f}x"
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.1f} | {:.2f} | {:.1f} | {:.1f} | {:.1f} | {} | {} |".format(
                input_len, batch, "parallel" if parallel else "serial",
                r["ttft"]["mean_ms"], r["prefill_latency"]["mean_ms"],
                r["prefill_tok_s"]["mean_ms"],
                r["decode_step_latency"]["mean_ms"], r["decode_tok_s"]["mean_ms"],
                r["end_to_end_tok_s"]["mean_ms"], peak, note, r["status"]))
    lines += ["", f"Generated: {data['environment']['timestamp_utc']}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--batch-sizes", type=_ints, default=[1, 4, 8])
    parser.add_argument("--input-lengths", type=_ints, default=[1024, 2048, 4096])
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    max_batch = max(args.batch_sizes)
    max_input = max(args.input_lengths)
    for parallel in (False, True):
        engine = LLMEngine(args.model,
            dtype="float16", tensor_parallel_size=1, enforce_eager=True,
            max_num_seqs=max_batch, max_model_len=max_input + args.output_len,
            max_num_batched_tokens=max_batch * (max_input + args.output_len),
            paged_prefill_q_tile_parallel=parallel)
        try:
            for batch in args.batch_sizes:
                for input_len in args.input_lengths:
                    seed = 20260716 + batch * 100000 + input_len
                    result = run_mode(engine, batch, input_len, args.output_len,
                                      parallel, args.warmup_runs, args.repeats, seed)
                    results.append(result)
                    print(f"{'parallel' if parallel else 'serial':>8} batch={batch} input={input_len} "
                          f"TTFT={result['ttft']['mean_ms']:.2f}ms "
                          f"prefill={result['prefill_latency']['mean_ms']:.2f}ms", flush=True)
        finally:
            engine.exit()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

    data = {
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "triton_version": triton.__version__,
            "model": args.model, "dtype": "float16",
            "tensor_parallel_size": 1, "cuda_graph": "disabled",
        },
        "command": " ".join(sys.argv), "parameters": vars(args), "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"qwen3_eager_q_tile_{stamp}.json"
    md_path = output_dir / f"qwen3_eager_q_tile_{stamp}.md"
    required = {
        "ttft", "prefill_latency", "prefill_tok_s", "decode_step_latency",
        "decode_tok_s", "end_to_end_tok_s", "peak_memory_allocated_bytes",
    }
    for result in data["results"]:
        missing = required.difference(result)
        if missing:
            raise RuntimeError(f"result schema missing fields: {sorted(missing)}")
        raw_required = {"ttft_ms", "prefill_tokens", "prefill_ms", "prefill_tok_s",
                        "decode_step_ms", "decode_tok_s", "end_to_end_tok_s"}
        for run in result["raw_runs"]:
            raw_missing = raw_required.difference(run)
            if raw_missing:
                raise RuntimeError(f"raw run schema missing fields: {sorted(raw_missing)}")
    json_text = json.dumps(data, indent=2) + "\n"
    markdown_text = markdown(data)
    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    json_tmp.write_text(json_text)
    md_tmp.write_text(markdown_text)
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
