# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:20:58+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/formal_input128_seqs16.jsonl`](formal_input128_seqs16.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 16 --input-len 128 --output-len 64 --warmup-runs 2 --runs 5 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/formal_input128_seqs16.jsonl --output-md docs/benchmarks/matrix/formal_input128_seqs16.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `16`
- Input tokens per sequence: `128`
- Output tokens per sequence: `64`
- Warmup runs: `2`
- Measured runs: `5`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234, 11235, 11236, 11237, 11238]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `4096`; max sequences: `16`; max batched tokens: `32768`
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
| nano_eager | eager | ok | 571.12 | 0.42 | 570.63 | 571.80 | 1.7908, 1.7945, 1.7932, 1.7930, 1.7933 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| nano_graph | graph | ok | 2104.61 | 2.86 | 2102.63 | 2109.22 | 0.4855, 0.4863, 0.4870, 0.4870, 0.4870 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| hf | hf | ok | 563.86 | 0.10 | 563.69 | 563.94 | 1.8158, 1.8160, 1.8161, 1.8159, 1.8166 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| vllm | vllm | ok | 2093.10 | 14.28 | 2068.53 | 2102.32 | 0.4894, 0.4873, 0.4950, 0.4871, 0.4875 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
