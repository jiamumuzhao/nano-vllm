"""Worker process for one offline generation benchmark engine."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_schema import deterministic_prompt_ids, environment_info, now_utc, set_seed, summarize_measurements, synchronize_cuda

RESULT_MARKER = "###BENCHMARK_RESULT_JSON###"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one offline generation benchmark engine")
    parser.add_argument("--engine", required=True, choices=("nano_eager", "nano_graph", "hf", "vllm"))
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
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
    return parser.parse_args()


def torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def warmup_seed(args: argparse.Namespace, warmup_idx: int) -> int:
    return args.seed + warmup_idx


def measured_seed(args: argparse.Namespace, run_idx: int) -> int:
    return args.seed + 10_000 + run_idx


def make_base_result(args: argparse.Namespace, prompt_info: dict[str, Any], initialization_seconds: float) -> dict[str, Any]:
    return {
        "record_type": "summary",
        "status": "ok",
        "created_at": now_utc(),
        "engine": args.engine,
        "mode": args.engine.replace("nano_", "") if args.engine.startswith("nano_") else args.engine,
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "temperature": args.temperature,
        "timed_region": "generation only after model/tokenizer initialization and warmup",
        "initialization_seconds": initialization_seconds,
        "workload": {
            "num_seqs": args.num_seqs,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "prompt": {
            "input_token_count_per_sequence": args.input_len,
            "num_sequences": args.num_seqs,
            "prompt_token_sha256": prompt_info["prompt_token_sha256"],
            "tokenizer_class": prompt_info["tokenizer_class"],
            "vocab_size": prompt_info["vocab_size"],
        },
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "fixed_output_length": True,
            "ignore_eos_or_equivalent": True,
            "top_k": "disabled",
            "top_p": 1.0,
            "is_greedy": False,
            "warmup_seeds": [warmup_seed(args, i) for i in range(args.warmup_runs)],
            "measured_seeds": [measured_seed(args, i) for i in range(args.runs)],
        },
        "environment": environment_info(),
        "measurements": [],
    }


def run_nano(args: argparse.Namespace, prompts: list[list[int]]) -> dict[str, Any]:
    from nanovllm import LLM, SamplingParams

    enforce_eager = args.engine == "nano_eager"
    params = [SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len) for _ in prompts]
    started = time.perf_counter()
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dtype=args.dtype,
    )
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - started
    measurements: list[dict[str, Any]] = []
    try:
        for warmup_idx in range(args.warmup_runs):
            seed = warmup_seed(args, warmup_idx)
            set_seed(seed)
            llm.generate(prompts, params, use_tqdm=False)
            synchronize_cuda()
        for run_idx in range(args.runs):
            seed = measured_seed(args, run_idx)
            set_seed(seed)
            synchronize_cuda()
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            synchronize_cuda()
            seconds = time.perf_counter() - started
            generated_tokens = sum(len(item["token_ids"]) for item in outputs)
            measurements.append({
                "run_index": run_idx,
                "seed": seed,
                "seconds": seconds,
                "generated_tokens": generated_tokens,
                "expected_generated_tokens": args.num_seqs * args.output_len,
                "output_tokens_per_second": generated_tokens / seconds,
                "all_outputs_fixed_length": all(len(item["token_ids"]) == args.output_len for item in outputs),
            })
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
    return {"initialization_seconds": initialization_seconds, "measurements": measurements}


def run_hf(args: argparse.Namespace, prompts: list[list[int]]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype(args.dtype))
    model.eval().to("cuda")
    input_ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids, device="cuda")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - started
    measurements: list[dict[str, Any]] = []
    generate_kwargs = {
        "max_new_tokens": args.output_len,
        "min_new_tokens": args.output_len,
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": 0,
        "top_p": 1.0,
        "eos_token_id": None,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }
    with torch.inference_mode():
        for warmup_idx in range(args.warmup_runs):
            seed = warmup_seed(args, warmup_idx)
            set_seed(seed)
            model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
            synchronize_cuda()
        for run_idx in range(args.runs):
            seed = measured_seed(args, run_idx)
            set_seed(seed)
            synchronize_cuda()
            started = time.perf_counter()
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
            synchronize_cuda()
            seconds = time.perf_counter() - started
            generated_tokens = int(outputs.shape[0] * max(0, outputs.shape[1] - input_ids.shape[1]))
            measurements.append({
                "run_index": run_idx,
                "seed": seed,
                "seconds": seconds,
                "generated_tokens": generated_tokens,
                "expected_generated_tokens": args.num_seqs * args.output_len,
                "output_tokens_per_second": generated_tokens / seconds,
                "all_outputs_fixed_length": generated_tokens == args.num_seqs * args.output_len,
            })
    return {
        "initialization_seconds": initialization_seconds,
        "measurements": measurements,
        "hf_generation_config": generate_kwargs | {"pad_token_id": int(pad_token_id) if pad_token_id is not None else None},
    }


def make_vllm_sampling_params(args: argparse.Namespace, seed: int):
    from vllm import SamplingParams as VllmSamplingParams

    return VllmSamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        ignore_eos=True,
        top_k=0,
        top_p=1.0,
        seed=seed,
    )


def run_vllm(args: argparse.Namespace, prompts: list[list[int]]) -> dict[str, Any]:
    from vllm import LLM

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype if args.dtype != "auto" else "auto",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=False,
    )
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - started
    measurements: list[dict[str, Any]] = []
    prompt_token_ids = [{"prompt_token_ids": row} for row in prompts]
    for warmup_idx in range(args.warmup_runs):
        seed = warmup_seed(args, warmup_idx)
        set_seed(seed)
        sampling_params = make_vllm_sampling_params(args, seed)
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        synchronize_cuda()
    for run_idx in range(args.runs):
        seed = measured_seed(args, run_idx)
        set_seed(seed)
        sampling_params = make_vllm_sampling_params(args, seed)
        synchronize_cuda()
        started = time.perf_counter()
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        synchronize_cuda()
        seconds = time.perf_counter() - started
        generated_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
        measurements.append({
            "run_index": run_idx,
            "seed": seed,
            "seconds": seconds,
            "generated_tokens": generated_tokens,
            "expected_generated_tokens": args.num_seqs * args.output_len,
            "output_tokens_per_second": generated_tokens / seconds,
            "all_outputs_fixed_length": all(len(item.outputs[0].token_ids) == args.output_len for item in outputs),
        })
    return {
        "initialization_seconds": initialization_seconds,
        "measurements": measurements,
        "vllm_generation_config": {
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "sampling_seed_supported": True,
            "sampling_seed_source": "vllm.SamplingParams(seed=...) per warmup/measured run",
            "top_k": 0,
            "top_p": 1.0,
            "ignore_eos": True,
            "min_tokens": args.output_len,
            "max_tokens": args.output_len,
        },
    }

def main() -> None:
    args = parse_args()
    if args.temperature <= 1e-10:
        raise SystemExit("temperature must be > 1e-10 to match Nano-vLLM SamplingParams")
    prompt_info = deterministic_prompt_ids(args.model, args.num_seqs, args.input_len, args.seed)
    prompts = prompt_info.pop("prompt_token_ids")
    if args.engine.startswith("nano_"):
        engine_result = run_nano(args, prompts)
    elif args.engine == "hf":
        engine_result = run_hf(args, prompts)
    elif args.engine == "vllm":
        engine_result = run_vllm(args, prompts)
    else:
        raise AssertionError(args.engine)
    result = make_base_result(args, prompt_info, engine_result["initialization_seconds"])
    result["measurements"] = engine_result["measurements"]
    if "hf_generation_config" in engine_result:
        result["hf_generation_config"] = engine_result["hf_generation_config"]
    if "vllm_generation_config" in engine_result:
        result["vllm_generation_config"] = engine_result["vllm_generation_config"]
    result["summary"] = summarize_measurements(result["measurements"])
    print(RESULT_MARKER)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
