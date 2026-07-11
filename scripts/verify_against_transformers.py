"""Compare Nano-vLLM sampled prefill logits against Hugging Face Qwen3."""

import argparse
import atexit
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import reset_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--max-abs-error", type=float, default=0.1)
    args = parser.parse_args()
    prompt = list(range(1, args.prompt_len + 1))
    llm = LLM(args.model, tensor_parallel_size=1, enforce_eager=True, dtype="float16", max_num_batched_tokens=args.prompt_len, max_model_len=args.prompt_len)
    try:
        llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=1))
        seqs, is_prefill = llm.scheduler.schedule()
        input_ids, positions = llm.model_runner.prepare_prefill(seqs)
        nano_logits = llm.model_runner.run_model(input_ids, positions, is_prefill).float()
        reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        with torch.inference_mode():
            reference_logits = reference(input_ids=torch.tensor([prompt], device="cuda"), position_ids=torch.arange(args.prompt_len, device="cuda").unsqueeze(0), use_cache=False).logits[0, -1:].float()
        max_abs_error = (nano_logits - reference_logits).abs().max().item()
        nano_top1 = nano_logits.argmax(dim=-1).item()
        reference_top1 = reference_logits.argmax(dim=-1).item()
        if max_abs_error > args.max_abs_error:
            raise AssertionError(f"max_abs_error={max_abs_error:.6f} exceeds threshold {args.max_abs_error}")
        if nano_top1 != reference_top1:
            raise AssertionError(f"top1 mismatch: nano={nano_top1}, reference={reference_top1}")
        print(f"PASS: {args.prompt_len} prefill tokens sampled logits; max_abs_error={max_abs_error:.6f}; top1={nano_top1}")
    finally:
        reset_context()
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
