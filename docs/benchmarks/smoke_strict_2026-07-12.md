# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T06:49:30+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/smoke_strict_2026-07-12.jsonl`](smoke_strict_2026-07-12.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 1 --input-len 16 --output-len 4 --warmup-runs 0 --runs 1 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 64 --max-num-seqs 1 --max-num-batched-tokens 64 --gpu-memory-utilization 0.3 --output-jsonl docs/benchmarks/smoke_strict_2026-07-12.jsonl --output-md docs/benchmarks/smoke_strict_2026-07-12.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `1`
- Input tokens per sequence: `16`
- Output tokens per sequence: `4`
- Warmup runs: `0`
- Measured runs: `1`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `64`; max sequences: `1`; max batched tokens: `64`
- Fixed output length: Nano-vLLM uses `ignore_eos=True`; Hugging Face uses `min_new_tokens=max_new_tokens` and `eos_token_id=None`; vLLM uses `ignore_eos=True` plus `min_tokens=max_tokens`.

## Environment

- Python: `/root/workspace/nano-vllm/.venv-vllm-bench/bin/python` / `3.13.9`
- PyTorch: `2.11.0+cu130`
- CUDA runtime: `13.0`
- Triton: `3.6.0`
- Transformers: `5.12.1`
- Git revision: `5fc5453e1245fbd31f3906ca6a618f8276690c1b`; dirty: `True`
- GPU 0: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11337531392` bytes
- GPU 1: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11347623936` bytes

## Results

| Engine | Mode | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Seeds | JSONL |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| nano_eager | eager | ok | 2.78 | 0.00 | 2.78 | 2.78 | 1.4375 | 11234 | `docs/benchmarks/smoke_strict_2026-07-12.jsonl` |
| nano_graph | graph | ok | 5.50 | 0.00 | 5.50 | 5.50 | 0.7269 | 11234 | `docs/benchmarks/smoke_strict_2026-07-12.jsonl` |
| hf | hf | ok | 11.98 | 0.00 | 11.98 | 11.98 | 0.3338 | 11234 | `docs/benchmarks/smoke_strict_2026-07-12.jsonl` |
| vllm | vllm | ok | 62.66 | 0.00 | 62.66 | 62.66 | 0.0638 | 11234 | `docs/benchmarks/smoke_strict_2026-07-12.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
