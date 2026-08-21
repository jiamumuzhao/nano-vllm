# Nano-vLLM Online AsyncEngine Serving Benchmark

This is a real `AsyncEngine.generate()` online serving benchmark with concurrent streaming consumers; it is not an offline `LLM.generate()` result.

Command: `/opt/anaconda3/bin/python scripts/benchmark_serving.py --model /root/huggingface/Qwen3-0.6B --concurrencies 1,4,8,16 --input-lens 2048 --output-lens 64 --warmup-runs 1 --runs 1 --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9 --output-jsonl docs/benchmarks/online_serving_2048_maxlen4096_2026-08-21.jsonl --output-md docs/benchmarks/online_serving_2048_maxlen4096_2026-08-21.md`

TTFT is request submission to first output. TPOT is the mean interval between adjacent output tokens for one request and is null for one-token completions. Percentiles use sorted linear interpolation at `(n-1)*p/100`. Prefill/decode tok/s are null because the current AsyncEngine does not expose an unambiguous phase boundary.

Environment: `{"cuda_available": true, "cuda_runtime": "13.0", "git": {"commit": "27f9a94be438410fafe223bdea4560543e2a8560", "is_dirty": false, "status_short": []}, "gpu_count": 2, "gpus": [{"capability": "7.5", "index": 0, "name": "NVIDIA GeForce RTX 2080 Ti", "total_memory_bytes": 11337531392}, {"capability": "7.5", "index": 1, "name": "NVIDIA GeForce RTX 2080 Ti", "total_memory_bytes": 11347623936}], "nvidia_smi": ["595.71.05, NVIDIA GeForce RTX 2080 Ti, 7.5, 11264 MiB", "595.71.05, NVIDIA GeForce RTX 2080 Ti, 7.5, 11264 MiB"], "platform": "Linux-6.8.0-124-generic-x86_64-with-glibc2.39", "python": "3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]", "python_executable": "/opt/anaconda3/bin/python", "torch": "2.11.0+cu130", "transformers": "5.12.1", "triton": "3.6.0"}`

KV preflight is a fast diagnostic, not a guarantee: `theoretical_minimum_utilization_for_one_kv_block` covers only estimated model bytes plus one KV block. The conservative action is to increase `gpu_memory_utilization` or reduce `max_model_len`/`max_num_seqs`; warmup workspace and allocator fragmentation can still cause `preflight passed; runtime allocation failed`.

| concurrency | input | output | status | TTFT p50/p95/p99 (s) | TPOT p50/p95/p99 (s) | E2E p50/p95/p99 (s) | output tok/s | KV peak | prefix hit/token-hit | failure |
|---:|---:|---:|---|---|---|---|---:|---:|---|---|
| 1 | 2048 | 64 | ok | 2.8353626020252705/2.8353626020252705/2.8353626020252705 | 0.01194237122341754/0.01194237122341754/0.01194237122341754 | 3.5877319891005754/3.5877319891005754/3.5877319891005754 | 17.83856770640341 | 132 | 0.0/0.0 |  |
| 4 | 2048 | 64 | ok | 10.003523507155478/10.003547127684579/10.003549770889803 | 0.014334829930689127/0.01433511389611614/0.014335138169043356 | 10.906620322726667/10.90662541212514/10.906626078430563 | 23.471857656250872 | 528 | 0.0/0.0 |  |
| 8 | 2048 | 64 | ok | 19.62013497436419/19.62016684045084/19.620173029312863 | 0.020719959252765256/0.020720494079536626/0.020720542570398678 | 20.925492407288402/20.925510262697934/20.925512000471354 | 24.467645099749365 | 1056 | 0.0/0.0 |  |
| 16 | 2048 | 64 | ok | 39.50152143742889/39.50156069919467/39.50157264843583 | 0.0387384366347558/0.03873949589961696/0.03873959095660774 | 41.94204294541851/41.942081385757774/41.94208492515609 | 24.414543303640983 | 2112 | 0.0/0.0 |  |

JSONL: `docs/benchmarks/online_serving_2048_maxlen4096_2026-08-21.jsonl`
