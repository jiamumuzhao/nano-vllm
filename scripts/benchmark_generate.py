"""Repeatable offline throughput benchmark. Outputs JSON for easy comparisons."""

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    prompts = [[(token + seq) % 10_000 for token in range(args.input_len)] for seq in range(args.num_seqs)]
    params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len) for _ in prompts]
    started = time.perf_counter()
    llm = LLM(args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, dtype="float16")
    synchronize()
    initialization_seconds = time.perf_counter() - started
    try:
        for _ in range(args.warmup_runs):
            llm.generate(prompts, params, use_tqdm=False)
        measurements = []
        tokens = args.num_seqs * args.output_len
        for _ in range(args.runs):
            synchronize()
            started = time.perf_counter()
            llm.generate(prompts, params, use_tqdm=False)
            synchronize()
            seconds = time.perf_counter() - started
            measurements.append({"seconds": seconds, "end_to_end_output_tokens_per_second": tokens / seconds})
        result = {"workload": vars(args), "timed_region": "full llm.generate call after warmup", "target_output_tokens": tokens, "initialization_seconds": initialization_seconds, "measurements": measurements, "mean_end_to_end_output_tokens_per_second": sum(x["end_to_end_output_tokens_per_second"] for x in measurements) / len(measurements), "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
