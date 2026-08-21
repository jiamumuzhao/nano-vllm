# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:05:50+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/formal_input128_seqs1.jsonl`](formal_input128_seqs1.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 1 --input-len 128 --output-len 64 --warmup-runs 2 --runs 5 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/formal_input128_seqs1.jsonl --output-md docs/benchmarks/matrix/formal_input128_seqs1.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `1`
- Input tokens per sequence: `128`
- Output tokens per sequence: `64`
- Warmup runs: `2`
- Measured runs: `5`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234, 11235, 11236, 11237, 11238]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `4096`; max sequences: `1`; max batched tokens: `32768`
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
| nano_eager | eager | ok | 42.44 | 0.07 | 42.35 | 42.54 | 1.5072, 1.5073, 1.5112, 1.5092, 1.5044 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| nano_graph | graph | ok | 251.98 | 0.05 | 251.93 | 252.05 | 0.2539, 0.2540, 0.2540, 0.2540, 0.2540 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| hf | hf | ok | 38.27 | 0.02 | 38.24 | 38.30 | 1.6716, 1.6716, 1.6738, 1.6712, 1.6724 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| vllm | vllm | ok | 277.07 | 0.15 | 276.80 | 277.15 | 0.2309, 0.2309, 0.2312, 0.2309, 0.2309 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
