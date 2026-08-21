"""Small eager W8A16 model benchmark; CUDA Graph is intentionally disabled."""
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
    return [int(x) for x in value.split(",") if x]


def _summary(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(torch.tensor(values).quantile(0.5).item()),
        "p95_ms": float(torch.tensor(values).quantile(0.95).item()),
    }


def _codegen_metadata():
    root = Path(__file__).resolve().parents[1] / "docs/benchmarks"
    files = sorted(root.glob("w8a16_tensorcore_codegen_*.json"))
    if not files:
        return {"implementation": "unverified", "evidence": None, "status": "missing"}
    try:
        data = json.loads(files[-1].read_text())
        return {"implementation": data.get("kernel_path", "unsupported"),
                "evidence": str(files[-1]), "status": data.get("status", "unknown")}
    except Exception:
        return {"implementation": "unverified", "evidence": str(files[-1]), "status": "unreadable"}


def _run_once(engine, prompts, output_len, seed):
    torch.manual_seed(seed)
    params = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, params)
    start = time.perf_counter()
    prefill_ms = 0.0
    prefill_tokens = 0
    decode_ms = []
    while not engine.is_finished():
        t = time.perf_counter()
        _, n, _ = engine.step_with_events()
        elapsed = (time.perf_counter() - t) * 1000
        if n > 0:
            prefill_ms += elapsed
            prefill_tokens += n
        else:
            decode_ms.append(elapsed)
    total_ms = (time.perf_counter() - start) * 1000
    decode_total = statistics.fsum(decode_ms)
    return {
        "prefill_tokens": prefill_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tokens * 1000 / prefill_ms,
        "decode_step_ms": statistics.fmean(decode_ms),
        "decode_tok_s": len(prompts) * output_len * 1000 / decode_total,
        "end_to_end_tok_s": len(prompts) * output_len * 1000 / total_ms,
    }


def run_mode(args, quantization):
    torch.cuda.empty_cache()
    engine = LLMEngine(
        args.model, dtype="float16", tensor_parallel_size=1,
        enforce_eager=True, quantization=quantization,
        max_num_seqs=max(args.batch_sizes),
        max_model_len=max(args.input_lengths) + args.output_len,
        max_num_batched_tokens=max(args.batch_sizes) * (max(args.input_lengths) + args.output_len),
    )
    load_memory = torch.cuda.memory_allocated()
    results = []
    try:
        for batch in args.batch_sizes:
            for input_len in args.input_lengths:
                prompts = [[1] * input_len for _ in range(batch)]
                for i in range(args.warmup_runs):
                    _run_once(engine, prompts, args.output_len, args.seed + i)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                runs = [_run_once(engine, prompts, args.output_len,
                                  args.seed + args.warmup_runs + i)
                        for i in range(args.repeats)]
                torch.cuda.synchronize()
                results.append({
                    "quantization": quantization,
                    "batch_size": batch,
                    "input_len": input_len,
                    "output_len": args.output_len,
                    "status": "ok",
                    "load_memory_allocated_bytes": load_memory,
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "prefill_latency": _summary([x["prefill_ms"] for x in runs]),
                    "prefill_tok_s": _summary([x["prefill_tok_s"] for x in runs]),
                    "decode_step_latency": _summary([x["decode_step_ms"] for x in runs]),
                    "decode_tok_s": _summary([x["decode_tok_s"] for x in runs]),
                    "end_to_end_tok_s": _summary([x["end_to_end_tok_s"] for x in runs]),
                    "raw_runs": runs,
                })
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--batch-sizes", type=_ints, default=[1])
    parser.add_argument("--input-lengths", type=_ints, default=[1024])
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = []
    for mode in ("none", "w8a16"):
        results.extend(run_mode(args, mode))
    codegen = _codegen_metadata()
    data = {
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "triton_version": triton.__version__,
            "model": args.model, "dtype": "float16",
            "tensor_parallel_size": 1, "cuda_graph": "disabled",
            "w8a16_kernel_implementation": codegen["implementation"],
            "w8a16_codegen_evidence": codegen["evidence"],
            "w8a16_codegen_status": codegen["status"],
        },
        "command": " ".join(sys.argv), "parameters": vars(args), "results": results,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp, mp = out / f"w8a16_{stamp}.json", out / f"w8a16_{stamp}.md"
    groups = {(r["batch_size"], r["input_len"]): {} for r in results}
    for r in results: groups[(r["batch_size"], r["input_len"])][r["quantization"]] = r
    lines = ["# W8A16 eager benchmark", "", "CUDA Graph disabled; activation/KV cache remain FP16.", "",
             "| Batch | Input | Mode | Load MB | Peak MB | Prefill ms | Prefill tok/s | Decode step ms | Decode tok/s | E2E tok/s |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (batch, length), modes in sorted(groups.items()):
        for mode in ("none", "w8a16"):
            r = modes[mode]
            lines.append(f"| {batch} | {length} | {mode} | {r['load_memory_allocated_bytes']/2**20:.1f} | {r['peak_memory_allocated_bytes']/2**20:.1f} | {r['prefill_latency']['mean_ms']:.2f} | {r['prefill_tok_s']['mean_ms']:.1f} | {r['decode_step_latency']['mean_ms']:.2f} | {r['decode_tok_s']['mean_ms']:.1f} | {r['end_to_end_tok_s']['mean_ms']:.1f} |")
    lines += ["", "Results are reported as measured; W8A16 is not assumed to accelerate every workload.", ""]
    jp.write_text(json.dumps(data, indent=2) + "\n")
    mp.write_text("\n".join(lines))
    print(f"JSON: {jp}\nMarkdown: {mp}")


if __name__ == "__main__":
    main()
