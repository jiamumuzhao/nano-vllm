# Correctness and Performance Baseline

Established on 2026-07-11 in `/root/workspace/nano-vllm`.

## Environment

- Host alias: `Ubuntu`
- Python: `/opt/anaconda3/bin/python` 3.13.9
- PyTorch: 2.11.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Triton: 3.6.0
- Transformers: 5.12.1
- GPUs: 2 x NVIDIA GeForce RTX 2080 Ti
- Model: `/root/huggingface/Qwen3-0.6B`

## Correctness Checks

Run the unit and kernel correctness suite:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python -m pytest -q
```

Current result:

```text
7 passed, 16 warnings in 5.11s
```

Run the Hugging Face alignment check:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python scripts/verify_against_transformers.py \
  --model /root/huggingface/Qwen3-0.6B \
  --prompt-len 16
```

Current result:

```text
PASS: 16 prefill tokens sampled logits; max_abs_error=0.066406; top1=17
```

The HF check compares the sampled prefill logits used by generation, not every intermediate token logit. The default threshold is `--max-abs-error 0.1`, with top-1 equality required.

## Performance Workload

The benchmark reports end-to-end output tokens per second for full `llm.generate()` calls after warmup. It also prints the engine's internal prefill/decode timing for reference.

Shared workload:

- Tensor parallel size: 1
- Sequences: 8
- Prompt length: 128
- Requested output length: 64
- Max model length: 512
- Max sequences: 8
- GPU memory utilization: 0.5
- Warmup runs: 1
- Measured runs: 3

### Eager

Command:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python scripts/benchmark_generate.py \
  --model /root/huggingface/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --num-seqs 8 \
  --max-num-seqs 8 \
  --input-len 128 \
  --output-len 64 \
  --warmup-runs 1 \
  --runs 3 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager
```

Current result:

```text
mean_end_to_end_output_tokens_per_second = 305.78361946551905
run_seconds = [1.6863401090959087, 1.6709820190444589, 1.6659718920709565]
engine_decode_printed = about 319.5 to 321.2 tok/s after warmup
```

### CUDA Graph

Command:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python scripts/benchmark_generate.py \
  --model /root/huggingface/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --num-seqs 8 \
  --max-num-seqs 8 \
  --input-len 128 \
  --output-len 64 \
  --warmup-runs 1 \
  --runs 3 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.5
```

Current result:

```text
mean_end_to_end_output_tokens_per_second = 1410.4785356279172
run_seconds = [0.36242160492111, 0.36311742500402033, 0.3634546120883897]
engine_decode_printed = about 1893.3 to 1900.2 tok/s after warmup
```

## Notes

- Use `/opt/anaconda3/bin/python`; `/usr/bin/python3` does not have the project test dependencies.
- `benchmark_generate.py` sets `max_num_seqs` explicitly so CUDA Graph capture matches the measured workload instead of the default 512 sequence capacity.
- JSON throughput is an end-to-end measurement around `llm.generate()`. The banner printed by the engine is the internal prefill/decode timing.

Additional evidence boundaries and the future benchmark protocol are documented in [benchmark.md](benchmark.md).
