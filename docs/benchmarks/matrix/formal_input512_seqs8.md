# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:31:06+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/formal_input512_seqs8.jsonl`](formal_input512_seqs8.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 8 --input-len 512 --output-len 64 --warmup-runs 2 --runs 5 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/formal_input512_seqs8.jsonl --output-md docs/benchmarks/matrix/formal_input512_seqs8.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `8`
- Input tokens per sequence: `512`
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
| nano_eager | eager | ok | 233.69 | 0.58 | 232.76 | 234.29 | 2.1917, 2.1906, 2.1877, 2.1853, 2.1997 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| nano_graph | graph | ok | 551.72 | 17.17 | 531.09 | 565.74 | 0.9050, 0.9053, 0.9124, 0.9569, 0.9641 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| hf | hf | ok | 246.43 | 0.38 | 245.84 | 246.87 | 2.0740, 2.0780, 2.0759, 2.0777, 2.0826 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| vllm | vllm | ok | 842.06 | 6.93 | 833.12 | 848.66 | 0.6121, 0.6146, 0.6033, 0.6040, 0.6064 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
