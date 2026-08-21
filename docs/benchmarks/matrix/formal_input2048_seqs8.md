# Unified Offline Generation Benchmark

Run timestamp: `2026-07-12T07:51:45+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/matrix/formal_input2048_seqs8.jsonl`](formal_input2048_seqs8.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 8 --input-len 2048 --output-len 64 --warmup-runs 2 --runs 5 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/matrix/formal_input2048_seqs8.jsonl --output-md docs/benchmarks/matrix/formal_input2048_seqs8.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `8`
- Input tokens per sequence: `2048`
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
| nano_eager | eager | ok | 136.25 | 0.23 | 135.94 | 136.43 | 3.7533, 3.7665, 3.7533, 3.7528, 3.7625 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| nano_graph | graph | ok | 174.06 | 0.51 | 173.39 | 174.59 | 2.9359, 2.9326, 2.9528, 2.9385, 2.9482 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| hf | hf | ok | 68.07 | 1.05 | 66.53 | 69.18 | 7.4014, 7.4512, 7.4860, 7.5785, 7.6960 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| vllm | vllm | ok | 310.18 | 1.78 | 307.97 | 312.84 | 1.6366, 1.6506, 1.6489, 1.6548, 1.6625 | 11234, 11235, 11236, 11237, 11238 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
