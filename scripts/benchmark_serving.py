"""Reproducible online AsyncEngine serving benchmark.

This intentionally drives AsyncEngine.generate() directly.  It is not an
offline LLM.generate() benchmark and does not include a synthetic server.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import shlex
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# Running `python scripts/benchmark_serving.py` places only `scripts/` on
# sys.path in this checkout; add the repository root for the local package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_schema import deterministic_prompt_ids, environment_info, set_seed, write_jsonl_record
from nanovllm.engine.async_engine import AsyncEngine
from nanovllm.sampling_params import SamplingParams


def parse_csv_ints(raw: str, name: str) -> list[int]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"{name} contains an empty value")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {item!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} values must be positive")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


def parse_ratio(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prefix-sharing-ratio must be a finite number in [0, 1]") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("prefix-sharing-ratio must be a finite number in [0, 1]")
    return value


def positive_cli_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def nonnegative_cli_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile with no third-party dependency."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timing_summary(values: list[float | None]) -> dict[str, float | None]:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {f"p{p}": percentile(usable, p) for p in (50, 95, 99)}


def compute_tpot(output_times_s: list[float]) -> float | None:
    """Return adjacent output-token interval; one token intentionally yields null."""
    if any(not math.isfinite(value) or value < 0 for value in output_times_s):
        raise ValueError("output timestamps must be finite and non-negative")
    if any(later < earlier for earlier, later in zip(output_times_s, output_times_s[1:])):
        raise ValueError("output timestamps must be non-decreasing")
    if len(output_times_s) <= 1:
        return None
    return sum(later - earlier for earlier, later in zip(output_times_s, output_times_s[1:])) / (len(output_times_s) - 1)


def output_throughput(output_tokens: int, seconds: float | None) -> float | None:
    if output_tokens <= 0 or seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None
    return output_tokens / seconds


def prompt_sha256(prompt_token_ids: list[list[int]]) -> str:
    payload = json.dumps(prompt_token_ids, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_prompts(model: str, concurrency: int, input_len: int, seed: int, sharing_ratio: float) -> dict[str, Any]:
    data = deterministic_prompt_ids(model, concurrency, input_len, seed)
    prompts = data["prompt_token_ids"]
    shared = int(input_len * sharing_ratio)
    if shared > 0 and concurrency > 1:
        prefix = prompts[0][:shared]
        for index in range(1, concurrency):
            prompts[index][:shared] = prefix
    data["prompt_token_ids"] = prompts
    # Recompute after shared-prefix substitution. The fingerprint must match
    # the exact complete prompt list submitted to AsyncEngine.generate().
    data["prompt_token_sha256"] = prompt_sha256(prompts)
    data["prefix_sharing_tokens"] = shared
    return data


def compute_global_timing(request_records: list[dict[str, Any]], workload_failed: bool = False) -> tuple[float | None, str | None]:
    """Compute first actual submission to last completion from absolute times."""
    if workload_failed:
        return None, "workload failed; global throughput is not computed"
    submitted = [record.get("submitted_at_absolute_s") for record in request_records]
    completed = [record.get("completed_at_absolute_s") for record in request_records]
    if not completed:
        return None, "no completed requests"
    if any(value is None or not math.isfinite(float(value)) for value in submitted + completed):
        return None, "submitted/completed absolute timestamps are missing or non-finite"
    first_submitted = min(float(value) for value in submitted)
    last_completed = max(float(value) for value in completed)
    seconds = last_completed - first_submitted
    if not math.isfinite(seconds) or seconds <= 0:
        return None, "last completion is not later than the first actual submission"
    return seconds, None


_DTYPE_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "auto": 2}


def estimate_model_bytes(model_config: Any, dtype: str) -> int:
    if dtype not in _DTYPE_ITEMSIZE:
        raise ValueError(f"unsupported dtype for KV preflight: {dtype!r}")
    hidden = int(getattr(model_config, "hidden_size", 0))
    layers = int(getattr(model_config, "num_hidden_layers", 0))
    heads = int(getattr(model_config, "num_attention_heads", 0))
    kv_heads = int(getattr(model_config, "num_key_value_heads", heads))
    intermediate = int(getattr(model_config, "intermediate_size", 0))
    vocab = int(getattr(model_config, "vocab_size", 0))
    if min(hidden, layers, heads, kv_heads, intermediate, vocab) <= 0:
        raise ValueError("model config lacks positive Qwen-compatible dimensions for preflight")
    parameters = vocab * hidden * 2
    parameters += layers * (hidden * (hidden + 2 * kv_heads * (hidden // heads)) + hidden * hidden)
    parameters += layers * (3 * hidden * intermediate)
    return int(parameters * _DTYPE_ITEMSIZE[dtype])


def kv_preflight(
    model_config: Any,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    free_bytes: int,
    total_bytes: int,
    max_model_len: int,
    max_num_seqs: int,
    kvcache_block_size: int = 16,
) -> dict[str, Any]:
    if dtype not in _DTYPE_ITEMSIZE:
        raise ValueError(f"dtype must be one of {sorted(_DTYPE_ITEMSIZE)}")
    if not isinstance(tensor_parallel_size, int) or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not math.isfinite(float(gpu_memory_utilization)) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        raise ValueError("free/total GPU memory values are invalid")
    if max_model_len <= 0 or max_num_seqs <= 0 or kvcache_block_size <= 0:
        raise ValueError("max_model_len, max_num_seqs and kvcache_block_size must be positive")
    kv_heads = int(getattr(model_config, "num_key_value_heads", 0))
    hidden = int(getattr(model_config, "hidden_size", 0))
    heads = int(getattr(model_config, "num_attention_heads", 0))
    layers = int(getattr(model_config, "num_hidden_layers", 0))
    head_dim = int(getattr(model_config, "head_dim", hidden // heads if heads else 0))
    local_kv_heads = kv_heads // tensor_parallel_size if tensor_parallel_size else 0
    if local_kv_heads <= 0 or min(layers, head_dim) <= 0:
        raise ValueError("model config/TP combination has no positive local KV head capacity")
    model_bytes = estimate_model_bytes(model_config, dtype)
    block_bytes = 2 * layers * kvcache_block_size * local_kv_heads * head_dim * _DTYPE_ITEMSIZE[dtype]
    target_bytes = int(total_bytes * gpu_memory_utilization)
    minimum_target_bytes = model_bytes + block_bytes
    minimum_utilization = minimum_target_bytes / total_bytes
    configured_blocks = max_num_seqs * max(1, math.ceil(max_model_len / kvcache_block_size))
    configured_kv_bytes = configured_blocks * block_bytes
    ok = target_bytes >= minimum_target_bytes and free_bytes >= block_bytes
    return {
        "ok": ok,
        "reason": None if ok else "configured memory budget cannot accommodate estimated model plus one KV block",
        "free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
        "gpu_memory_utilization": float(gpu_memory_utilization), "target_bytes": target_bytes,
        "estimated_model_bytes": model_bytes, "estimated_kv_block_bytes": block_bytes,
        "estimated_configured_kv_bytes": configured_kv_bytes,
        "theoretical_minimum_utilization_for_one_kv_block": minimum_utilization,
        "recommended_action": (
            "increase gpu_memory_utilization or reduce max_model_len/max_num_seqs; "
            "the theoretical bound is not a runtime allocation guarantee"
        ),
        "runtime_allocation_note": (
            "preflight only rejects an obvious model-plus-one-block budget failure; "
            "warmup workspace and allocator fragmentation can still cause runtime failure"
        ),
        "verified_smoke_configuration": {
            "gpu": "NVIDIA GeForce RTX 2080 Ti",
            "model": "/root/huggingface/Qwen3-0.6B",
            "tensor_parallel_size": 1,
            "concurrency": 1,
            "input_len": 32,
            "output_len": 4,
            "gpu_memory_utilization": 0.9,
            "status": "observed current-environment smoke only",
        },
        "tensor_parallel_size": tensor_parallel_size, "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs, "kvcache_block_size": kvcache_block_size,
        "dtype": dtype, "local_kv_heads": local_kv_heads, "head_dim": head_dim,
    }


def _cuda_memory() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                    "cuda_memory_reason": "CUDA is not available"}
        return {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cuda_memory_reason": None,
        }
    except Exception as exc:
        return {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                "cuda_memory_reason": repr(exc)}


def _cuda_mem_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"runtime_free_bytes": None, "runtime_total_bytes": None,
                    "runtime_memory_reason": "CUDA is not available"}
        free, total = torch.cuda.mem_get_info()
        return {"runtime_free_bytes": int(free), "runtime_total_bytes": int(total),
                "runtime_memory_reason": None}
    except Exception as exc:
        return {"runtime_free_bytes": None, "runtime_total_bytes": None,
                "runtime_memory_reason": repr(exc)}


def _cleanup_diagnostics(engine: AsyncEngine) -> dict[str, Any]:
    metrics = engine.get_metrics_snapshot()
    scheduler = getattr(engine.engine, "scheduler", None)
    workers = getattr(engine.engine, "ps", [])
    return {
        "active_requests": metrics["active_requests"],
        "scheduler_used_kv_blocks": metrics["scheduler"]["kv_blocks_used"],
        "tp_workers_alive": [worker.is_alive() for worker in workers],
        "cleanup_ok": (
            metrics["active_requests"] == 0
            and metrics["scheduler"]["kv_blocks_used"] == 0
            and not any(worker.is_alive() for worker in workers)
        ),
    }


async def run_once(
    args: argparse.Namespace,
    prompts: list[list[int]],
    run_index: int,
    warmup: bool,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine: AsyncEngine | None = None
    started = time.perf_counter()
    request_records: list[dict[str, Any]] = []
    failure: str | None = None
    metrics: dict[str, Any] | None = None
    cleanup: dict[str, Any] = {}
    runtime_memory = _cuda_mem_info()
    try:
        set_seed(args.seed + run_index)
        engine = AsyncEngine(
            args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            max_queue_size=args.max_queue_size,
            request_timeout_s=args.request_timeout_s,
            stream_queue_size=args.stream_queue_size,
        )
        await engine.start()
        barrier = asyncio.Event()
        batch_start = time.perf_counter()

        async def consume(index: int, prompt: list[int]):
            await barrier.wait()
            submitted = time.perf_counter()
            output_times: list[float] = []
            token_count = 0
            first_output = None
            finished_at = None
            finish_reason = None
            try:
                async for output in engine.generate(
                    prompt, SamplingParams(temperature=1e-5, max_tokens=args.output_lens[0], ignore_eos=True),
                    request_id=f"bench-{run_index}-{index}",
                ):
                    now = time.perf_counter()
                    count = len(output.delta_token_ids)
                    if count:
                        if first_output is None:
                            first_output = now
                        output_times.extend([now - submitted] * count)
                        token_count += count
                    if output.finished:
                        finished_at = now
                        finish_reason = output.finish_reason
            except Exception as exc:
                return {
                    "request_index": index, "submitted_at_s": submitted - batch_start,
                    "submitted_at_absolute_s": submitted,
                    "first_token_at_s": None, "output_token_times_s": output_times,
                    "completed_at_s": None, "output_tokens": token_count,
                    "completed_at_absolute_s": None,
                    "finish_reason": "error", "error": traceback.format_exc(),
                }
            tpot = compute_tpot(output_times)
            return {
                "request_index": index,
                "submitted_at_s": submitted - batch_start,
                "submitted_at_absolute_s": submitted,
                "first_token_at_s": first_output - submitted if first_output is not None else None,
                "output_token_times_s": output_times,
                "completed_at_s": finished_at - batch_start if finished_at is not None else None,
                "completed_at_absolute_s": finished_at if finished_at is not None else None,
                "output_tokens": token_count,
                "finish_reason": finish_reason,
                "ttft_s": first_output - submitted if first_output is not None else None,
                "tpot_s": tpot,
                "e2e_latency_s": finished_at - submitted if finished_at is not None else None,
            }

        tasks = [asyncio.create_task(consume(index, prompt)) for index, prompt in enumerate(prompts)]
        barrier.set()
        request_records = await asyncio.gather(*tasks)
        metrics = engine.get_metrics_snapshot()
        if any(record.get("error") or record.get("finish_reason") != "stop" for record in request_records):
            failure = "one or more requests did not finish with stop: " + repr(request_records)
        if not request_records or any(record["output_tokens"] <= 0 for record in request_records):
            failure = failure or "incomplete request output; no throughput was computed"
    except Exception as exc:
        detail = traceback.format_exc()
        if preflight and preflight.get("ok") and "allocate_kv_cache" in detail:
            failure = "preflight passed; runtime allocation failed\n" + detail
        else:
            failure = detail
        runtime_memory = _cuda_mem_info()
    finally:
        if engine is not None:
            try:
                await engine.shutdown()
            except Exception as exc:
                failure = failure or traceback.format_exc()
            cleanup = _cleanup_diagnostics(engine)
            if not cleanup.get("cleanup_ok"):
                failure = failure or f"cleanup failed: {cleanup}"
            if metrics is None:
                metrics = engine.get_metrics_snapshot()
            # Each warmup/measured run has an independent Engine lifecycle.
            # Release Python model references and cached CUDA blocks before the
            # next run; otherwise the allocator can make a valid next engine
            # appear to have no KV capacity.
            del engine
            engine = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    if metrics is None and engine is not None:
        metrics = engine.get_metrics_snapshot()
    output_tokens = sum(record.get("output_tokens", 0) for record in request_records)
    global_seconds, global_failure_reason = compute_global_timing(request_records, workload_failed=bool(failure))
    record = {
        "record_type": "serving_summary",
        "status": "failed" if failure else "ok",
        "warmup": warmup,
        "run_index": run_index,
        "concurrency": len(prompts),
        "input_len": len(prompts[0]) if prompts else 0,
        "output_len": args.output_lens[0],
        "request_records": request_records,
        "ttft_s": timing_summary([item.get("ttft_s") for item in request_records]),
        "tpot_s": timing_summary([item.get("tpot_s") for item in request_records]),
        "e2e_latency_s": timing_summary([item.get("e2e_latency_s") for item in request_records]),
        "global_end_to_end_output_tokens": output_tokens if not failure else None,
        "global_end_to_end_seconds": global_seconds if not failure else None,
        "global_output_tokens_per_second": output_throughput(output_tokens, global_seconds) if not failure else None,
        "global_timing_failure_reason": global_failure_reason,
        "prefill_tok_s": None,
        "decode_tok_s": None,
        "phase_metrics_reason": "AsyncEngine output events do not expose an unambiguous prefill/decode wall-time boundary",
        "metrics": metrics,
        "cleanup": cleanup,
        "memory": _cuda_memory(),
        "runtime_memory": runtime_memory,
        "failure": failure,
        "preflight": preflight,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return record


def markdown(records: list[dict[str, Any]], args: argparse.Namespace, env: dict[str, Any]) -> str:
    lines = [
        "# Nano-vLLM Online AsyncEngine Serving Benchmark",
        "",
        "This is a real `AsyncEngine.generate()` online serving benchmark with concurrent streaming consumers; it is not an offline `LLM.generate()` result.",
        "",
        f"Command: `{shlex.join(args._command)}`",
        "",
        "TTFT is request submission to first output. TPOT is the mean interval between adjacent output tokens for one request and is null for one-token completions. Percentiles use sorted linear interpolation at `(n-1)*p/100`. Prefill/decode tok/s are null because the current AsyncEngine does not expose an unambiguous phase boundary.",
        "",
        f"Environment: `{json.dumps(env, sort_keys=True)}`",
        "",
        "KV preflight is a fast diagnostic, not a guarantee: `theoretical_minimum_utilization_for_one_kv_block` covers only estimated model bytes plus one KV block. The conservative action is to increase `gpu_memory_utilization` or reduce `max_model_len`/`max_num_seqs`; warmup workspace and allocator fragmentation can still cause `preflight passed; runtime allocation failed`.",
        "",
        "| concurrency | input | output | status | TTFT p50/p95/p99 (s) | TPOT p50/p95/p99 (s) | E2E p50/p95/p99 (s) | output tok/s | KV peak | prefix hit/token-hit | failure |",
        "|---:|---:|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for record in records:
        metric = record.get("metrics") or {}
        scheduler = metric.get("scheduler", {})
        lines.append(
            f"| {record['concurrency']} | {record['input_len']} | {record['output_len']} | {record['status']} | "
            f"{record['ttft_s']['p50']}/{record['ttft_s']['p95']}/{record['ttft_s']['p99']} | "
            f"{record['tpot_s']['p50']}/{record['tpot_s']['p95']}/{record['tpot_s']['p99']} | "
            f"{record['e2e_latency_s']['p50']}/{record['e2e_latency_s']['p95']}/{record['e2e_latency_s']['p99']} | "
            f"{record['global_output_tokens_per_second']} | {scheduler.get('kv_blocks_peak_used')} | "
            f"{scheduler.get('prefix_cache_hit_rate')}/{scheduler.get('prefix_cache_token_hit_rate')} | {record.get('failure') or ''} |"
        )
    lines += ["", f"JSONL: `{args.output_jsonl}`", ""]
    return "\n".join(lines)


def run_kv_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Read CUDA/model config only; do not construct ModelRunner or allocate tensors."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "reason": "CUDA is not available", "cuda_available": False}
        free, total = torch.cuda.mem_get_info()
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model)
        result = kv_preflight(
            model_config, args.dtype, args.tensor_parallel_size, args.gpu_memory_utilization,
            int(free), int(total), args.max_model_len, args.max_num_seqs,
            int(getattr(model_config, "kvcache_block_size", 16)),
        )
        result["cuda_available"] = True
        result["gpu_index"] = torch.cuda.current_device()
        return result
    except Exception as exc:
        return {"ok": False, "reason": f"preflight error: {type(exc).__name__}: {exc}",
                "diagnostic": traceback.format_exc()}


def preflight_failure_record(args: argparse.Namespace, preflight: dict[str, Any], concurrency: int, input_len: int, output_len: int) -> dict[str, Any]:
    return {
        "record_type": "serving_summary", "status": "failed", "warmup": False,
        "run_index": 0, "concurrency": concurrency, "input_len": input_len,
        "output_len": output_len, "request_records": [],
        "ttft_s": timing_summary([]), "tpot_s": timing_summary([]),
        "e2e_latency_s": timing_summary([]), "global_end_to_end_output_tokens": None,
        "global_end_to_end_seconds": None, "global_output_tokens_per_second": None,
        "global_timing_failure_reason": "preflight failed; no request was submitted",
        "prefill_tok_s": None, "decode_tok_s": None,
        "phase_metrics_reason": "preflight failed before AsyncEngine creation",
        "metrics": None, "cleanup": {"cleanup_ok": True, "not_started": True},
        "memory": _cuda_memory(), "runtime_memory": _cuda_mem_info(),
        "failure": f"KV preflight failed: {preflight.get('reason')}; diagnostics={preflight}",
        "preflight": preflight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark online AsyncEngine serving with reproducible token-id prompts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrencies", default="1,8,32,128")
    parser.add_argument("--input-lens", default="128,1024,4096,8192")
    parser.add_argument("--output-lens", default="32,128,512")
    parser.add_argument("--runs", type=positive_cli_int, default=3)
    parser.add_argument("--warmup-runs", type=nonnegative_cli_int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=positive_cli_int, default=8192)
    parser.add_argument("--max-num-seqs", type=positive_cli_int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=positive_cli_int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-queue-size", type=positive_cli_int, default=256)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--stream-queue-size", type=positive_cli_int, default=16)
    parser.add_argument("--prefix-sharing-ratio", type=parse_ratio, default=0.0)
    parser.add_argument("--output-jsonl", default="docs/benchmarks/serving.jsonl")
    parser.add_argument("--output-md", default="docs/benchmarks/serving.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.concurrencies = parse_csv_ints(args.concurrencies, "concurrencies")
        args.input_lens = parse_csv_ints(args.input_lens, "input-lens")
        args.output_lens = parse_csv_ints(args.output_lens, "output-lens")
        if args.runs <= 0 or args.warmup_runs < 0:
            raise ValueError("runs must be positive and warmup-runs must be non-negative")
        if args.max_model_len <= 0 or args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
            raise ValueError("model and scheduler capacities must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    args._command = [sys.executable, "scripts/benchmark_serving.py", *(argv if argv is not None else sys.argv[1:])]
    env = environment_info()
    preflight = run_kv_preflight(args)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []

    async def execute():
        output_lens = list(args.output_lens)
        for concurrency in args.concurrencies:
            for input_len in args.input_lens:
                for output_len in output_lens:
                    args.output_lens = [output_len]
                    prompts_data = make_prompts(args.model, concurrency, input_len, args.seed, args.prefix_sharing_ratio)
                    prompts = prompts_data["prompt_token_ids"]
                    for warmup_index in range(args.warmup_runs):
                        await run_once(args, prompts, -(warmup_index + 1), True, preflight)
                    for run_index in range(args.runs):
                        record = await run_once(args, prompts, run_index, False, preflight)
                        record.update({"prompt_token_sha256": prompts_data["prompt_token_sha256"],
                                       "prefix_sharing_tokens": prompts_data["prefix_sharing_tokens"],
                                       "environment": env,
                                       "command": args._command,
                                       "model": args.model,
                                       "dtype": args.dtype,
                                       "tensor_parallel_size": args.tensor_parallel_size,
                                       "max_model_len": args.max_model_len,
                                       "max_num_seqs": args.max_num_seqs,
                                       "max_num_batched_tokens": args.max_num_batched_tokens})
                        records.append(record)
                        write_jsonl_record(output_jsonl, record)

    if not preflight.get("ok"):
        for concurrency in args.concurrencies:
            for input_len in args.input_lens:
                for output_len in args.output_lens:
                    record = preflight_failure_record(args, preflight, concurrency, input_len, output_len)
                    record.update({"environment": env, "command": args._command, "model": args.model, "dtype": args.dtype,
                                   "tensor_parallel_size": args.tensor_parallel_size})
                    records.append(record)
                    write_jsonl_record(output_jsonl, record)
    else:
        try:
            asyncio.run(execute())
        except Exception as exc:
            record = {"record_type": "serving_summary", "status": "failed", "failure": traceback.format_exc(),
                      "preflight": preflight}
            records.append(record)
            write_jsonl_record(output_jsonl, record)
    md = markdown(records, args, env)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_md.parent, delete=False) as handle:
        handle.write(md)
        temporary = Path(handle.name)
    temporary.replace(output_md)
    return 1 if any(record.get("status") == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
