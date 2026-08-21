# Strict Controlled Offline Generation Benchmark - 2026-07-12

Run timestamp: `2026-07-12T06:51:53+00:00`

Raw JSONL: [`/root/workspace/nano-vllm/docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl`](offline_comparison_strict_2026-07-12.jsonl)

## Command

```bash
/root/workspace/nano-vllm/.venv-vllm-bench/bin/python scripts/benchmark_offline.py --engines nano_eager,nano_graph,hf,vllm --model /root/huggingface/Qwen3-0.6B --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 3 --runs 10 --dtype float16 --seed 1234 --temperature 1.0 --tensor-parallel-size 1 --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.5 --output-jsonl docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl --output-md docs/benchmarks/offline_comparison_strict_2026-07-12.md
```

## Workload

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Sequences: `8`
- Input tokens per sequence: `128`
- Output tokens per sequence: `64`
- Warmup runs: `3`
- Measured runs: `10`
- Dtype: `float16`
- Base seed: `1234`
- Measured run seeds: `[11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243]`
- Temperature: `1.0`; stochastic sampling, not greedy
- Max model length: `512`; max sequences: `8`; max batched tokens: `1024`
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
| nano_eager | eager | ok | 311.09 | 1.66 | 308.17 | 312.97 | 1.6370, 1.6434, 1.6448, 1.6603, 1.6419, 1.6451, 1.6396, 1.6495, 1.6614, 1.6360 | 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243 | `docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl` |
| nano_graph | graph | ok | 1412.67 | 2.27 | 1409.66 | 1417.26 | 0.3613, 0.3627, 0.3628, 0.3632, 0.3631, 0.3625, 0.3621, 0.3625, 0.3621, 0.3620 | 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243 | `docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl` |
| hf | hf | ok | 302.23 | 0.25 | 301.92 | 302.55 | 1.6924, 1.6936, 1.6956, 1.6934, 1.6923, 1.6948, 1.6926, 1.6956, 1.6949, 1.6958 | 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243 | `docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl` |
| vLLM | vllm | ok | 1394.37 | 15.62 | 1376.34 | 1413.27 | 0.3720, 0.3711, 0.3693, 0.3696, 0.3623, 0.3623, 0.3624, 0.3695, 0.3705, 0.3632 | 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243 | `docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl` |

## vLLM runner

The vLLM row was run by a project-local venv runner created with `--system-site-packages`, reusing the preinstalled `vllm==0.24.0` package from the Conda environment. This avoids adding vLLM to project dependencies, but it is not a fully independent dependency environment.

## Fairness notes

- All engines receive the same deterministic prompt token IDs generated once from the same tokenizer and seed.
- Timing excludes model/tokenizer initialization and warmup; it covers generation only with CUDA synchronization before and after each measured run.
- Sampling is stochastic with `temperature > 0`, not greedy. Top-k is disabled and top-p is set to 1.0 for Hugging Face and vLLM.
- vLLM `SamplingParams(seed=...)` is set per warmup/measured run; run records include the effective measured seed.
- Nano-vLLM, Hugging Face, and vLLM have different implementations and batching internals; this is a controlled offline generation baseline, not an online serving latency benchmark.
