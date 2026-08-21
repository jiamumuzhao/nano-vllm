# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:13:22+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/formal_input128_seqs8.jsonl`](formal_input128_seqs8.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 2 --runs 5 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/formal_input128_seqs8.jsonl --output-md docs/benchmarks/matrix/formal_input128_seqs8.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `8`
- Input tokens per sequence: `128`
- Output tokens per sequence: `64`
- Warmup runs: `2`
- Measured runs: `5`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234, 11235, 11236, 11237, 11238]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `4096`; max sequences: `8`; max batched tokens: `32768`
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
| nano_eager | eager | ok | 298.60 | 1.41 | 296.81 | 300.33 | 1.7106, 1.7123, 1.7250, 1.7208, 1.7048 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| nano_graph | graph | ok | 1405.57 | 1.36 | 1404.27 | 1407.46 | 0.3646, 0.3640, 0.3646, 0.3638, 0.3644 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| hf | hf | ok | 291.08 | 0.12 | 290.89 | 291.19 | 1.7585, 1.7583, 1.7588, 1.7591, 1.7601 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| vllm | vllm | ok | 1393.55 | 14.99 | 1376.45 | 1405.76 | 0.3648, 0.3642, 0.3716, 0.3720, 0.3646 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
