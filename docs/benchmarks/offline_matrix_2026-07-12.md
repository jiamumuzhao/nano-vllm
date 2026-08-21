# Offline Workload Matrix Benchmark - 2026-07-12

This is a controlled 3x3 offline generation workload matrix. It does not measure TTFT, TPOT, P95/P99 latency, dynamic request arrivals, streaming behavior, or service quality.

Raw data:

- Summary JSONL: [offline_matrix_2026-07-12.jsonl](offline_matrix_2026-07-12.jsonl)
- Verification JSON: [offline_matrix_2026-07-12.verify.json](offline_matrix_2026-07-12.verify.json)
- Per-workload files: [matrix/](matrix/)

## Environment

- Python: `/root/workspace/nano-vllm/.venv-vllm-bench/bin/python`
- Git revision: `5fc5453e1245fbd31f3906ca6a618f8276690c1b`; dirty: `True`
- PyTorch: `2.11.0+cu130`; CUDA runtime: `13.0`; Triton: `3.6.0`; Transformers: `5.12.1`
- GPU 0: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11337531392` bytes
- GPU 1: `NVIDIA GeForce RTX 2080 Ti`, capability `7.5`, memory `11347623936` bytes

## Fixed Protocol

- Model: `/root/huggingface/Qwen3-0.6B`
- Engines: `nano_eager,nano_graph,hf,vllm`
- Matrix: `num_seqs=1,8,16` x `input_len=128,512,2048`
- Output length: `64`; warmup runs: `2`; measured runs: `5`
- Dtype: `float16`; tensor parallel size: `1`; temperature: `1.0`; seed: `1234`
- Max model length: `4096`; max batched tokens: `32768`; max num seqs equals workload `num_seqs`
- Fixed output length, deterministic prompt token IDs, and measured seeds are inherited from `scripts/benchmark_offline.py`.

## Preflight Status

| input_len | num_seqs | status | failure |
| ---: | ---: | --- | --- |
| 128 | 1 | ok | - |
| 128 | 8 | ok | - |
| 128 | 16 | ok | - |
| 512 | 1 | ok | - |
| 512 | 8 | ok | - |
| 512 | 16 | ok | - |
| 2048 | 1 | ok | - |
| 2048 | 8 | ok | - |
| 2048 | 16 | failed-engine | hf: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 10.56 GiB of which 3.49 GiB is free. Including non-PyTorch |

## Workload Results

### input_len=128, num_seqs=1

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 42.44 | 0.07 | 42.35 | 42.54 | 1.5072, 1.5073, 1.5112, 1.5092, 1.5044 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| nano_graph | ok | 251.98 | 0.05 | 251.93 | 252.05 | 0.2539, 0.2540, 0.2540, 0.2540, 0.2540 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| hf | ok | 38.27 | 0.02 | 38.24 | 38.30 | 1.6716, 1.6716, 1.6738, 1.6712, 1.6724 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |
| vllm | ok | 277.07 | 0.15 | 276.80 | 277.15 | 0.2309, 0.2309, 0.2312, 0.2309, 0.2309 | `docs/benchmarks/matrix/formal_input128_seqs1.jsonl` |

- Nano Graph / vLLM: `0.909`
- Nano eager / HF: `1.109`

### input_len=128, num_seqs=8

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 298.60 | 1.41 | 296.81 | 300.33 | 1.7106, 1.7123, 1.7250, 1.7208, 1.7048 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| nano_graph | ok | 1405.57 | 1.36 | 1404.27 | 1407.46 | 0.3646, 0.3640, 0.3646, 0.3638, 0.3644 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| hf | ok | 291.08 | 0.12 | 290.89 | 291.19 | 1.7585, 1.7583, 1.7588, 1.7591, 1.7601 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |
| vllm | ok | 1393.55 | 14.99 | 1376.45 | 1405.76 | 0.3648, 0.3642, 0.3716, 0.3720, 0.3646 | `docs/benchmarks/matrix/formal_input128_seqs8.jsonl` |

- Nano Graph / vLLM: `1.009`
- Nano eager / HF: `1.026`

### input_len=128, num_seqs=16

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 571.12 | 0.42 | 570.63 | 571.80 | 1.7908, 1.7945, 1.7932, 1.7930, 1.7933 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| nano_graph | ok | 2104.61 | 2.86 | 2102.63 | 2109.22 | 0.4855, 0.4863, 0.4870, 0.4870, 0.4870 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| hf | ok | 563.86 | 0.10 | 563.69 | 563.94 | 1.8158, 1.8160, 1.8161, 1.8159, 1.8166 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |
| vllm | ok | 2093.10 | 14.28 | 2068.53 | 2102.32 | 0.4894, 0.4873, 0.4950, 0.4871, 0.4875 | `docs/benchmarks/matrix/formal_input128_seqs16.jsonl` |

- Nano Graph / vLLM: `1.005`
- Nano eager / HF: `1.013`

### input_len=512, num_seqs=1

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 40.08 | 0.34 | 39.54 | 40.47 | 1.5946, 1.5815, 1.6186, 1.5931, 1.5960 | `docs/benchmarks/matrix/formal_input512_seqs1.jsonl` |
| nano_graph | ok | 171.83 | 0.03 | 171.78 | 171.85 | 0.3726, 0.3725, 0.3725, 0.3724, 0.3725 | `docs/benchmarks/matrix/formal_input512_seqs1.jsonl` |
| hf | ok | 37.30 | 0.09 | 37.17 | 37.42 | 1.7163, 1.7154, 1.7157, 1.7218, 1.7103 | `docs/benchmarks/matrix/formal_input512_seqs1.jsonl` |
| vllm | ok | 240.92 | 0.23 | 240.71 | 241.20 | 0.2658, 0.2659, 0.2654, 0.2653, 0.2657 | `docs/benchmarks/matrix/formal_input512_seqs1.jsonl` |

- Nano Graph / vLLM: `0.713`
- Nano eager / HF: `1.075`

### input_len=512, num_seqs=8

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 233.69 | 0.58 | 232.76 | 234.29 | 2.1917, 2.1906, 2.1877, 2.1853, 2.1997 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| nano_graph | ok | 551.72 | 17.17 | 531.09 | 565.74 | 0.9050, 0.9053, 0.9124, 0.9569, 0.9641 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| hf | ok | 246.43 | 0.38 | 245.84 | 246.87 | 2.0740, 2.0780, 2.0759, 2.0777, 2.0826 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |
| vllm | ok | 842.06 | 6.93 | 833.12 | 848.66 | 0.6121, 0.6146, 0.6033, 0.6040, 0.6064 | `docs/benchmarks/matrix/formal_input512_seqs8.jsonl` |

- Nano Graph / vLLM: `0.655`
- Nano eager / HF: `0.948`

### input_len=512, num_seqs=16

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 368.87 | 1.01 | 367.71 | 369.93 | 2.7834, 2.7728, 2.7848, 2.7711, 2.7681 | `docs/benchmarks/matrix/formal_input512_seqs16.jsonl` |
| nano_graph | ok | 634.72 | 12.02 | 623.45 | 652.38 | 1.5696, 1.5991, 1.6179, 1.6397, 1.6425 | `docs/benchmarks/matrix/formal_input512_seqs16.jsonl` |
| hf | ok | 298.79 | 2.15 | 297.02 | 302.28 | 3.3875, 3.4204, 3.4432, 3.4381, 3.4476 | `docs/benchmarks/matrix/formal_input512_seqs16.jsonl` |
| vllm | ok | 1036.08 | 6.43 | 1031.42 | 1046.54 | 0.9865, 0.9785, 0.9916, 0.9928, 0.9925 | `docs/benchmarks/matrix/formal_input512_seqs16.jsonl` |

- Nano Graph / vLLM: `0.613`
- Nano eager / HF: `1.235`

### input_len=2048, num_seqs=1

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 35.44 | 0.06 | 35.36 | 35.50 | 1.8098, 1.8030, 1.8043, 1.8076, 1.8037 | `docs/benchmarks/matrix/formal_input2048_seqs1.jsonl` |
| nano_graph | ok | 79.05 | 0.02 | 79.01 | 79.06 | 0.8100, 0.8095, 0.8095, 0.8096, 0.8096 | `docs/benchmarks/matrix/formal_input2048_seqs1.jsonl` |
| hf | ok | 33.42 | 0.03 | 33.40 | 33.47 | 1.9119, 1.9152, 1.9163, 1.9153, 1.9160 | `docs/benchmarks/matrix/formal_input2048_seqs1.jsonl` |
| vllm | ok | 156.41 | 0.03 | 156.37 | 156.44 | 0.4091, 0.4091, 0.4092, 0.4092, 0.4093 | `docs/benchmarks/matrix/formal_input2048_seqs1.jsonl` |

- Nano Graph / vLLM: `0.505`
- Nano eager / HF: `1.061`

### input_len=2048, num_seqs=8

| Engine | Status | Mean tok/s | Stddev | Min | Max | Run seconds | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nano_eager | ok | 136.25 | 0.23 | 135.94 | 136.43 | 3.7533, 3.7665, 3.7533, 3.7528, 3.7625 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| nano_graph | ok | 174.06 | 0.51 | 173.39 | 174.59 | 2.9359, 2.9326, 2.9528, 2.9385, 2.9482 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| hf | ok | 68.07 | 1.05 | 66.53 | 69.18 | 7.4014, 7.4512, 7.4860, 7.5785, 7.6960 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |
| vllm | ok | 310.18 | 1.78 | 307.97 | 312.84 | 1.6366, 1.6506, 1.6489, 1.6548, 1.6625 | `docs/benchmarks/matrix/formal_input2048_seqs8.jsonl` |

- Nano Graph / vLLM: `0.561`
- Nano eager / HF: `2.002`

### input_len=2048, num_seqs=16

Formal four-engine comparison was skipped because preflight did not pass for every engine.


## Conclusion Table

| input_len | num_seqs | Nano Graph tok/s | vLLM tok/s | Faster | Nano Graph / vLLM | Nano eager / HF |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 128 | 1 | 251.98 | 277.07 | vLLM | 0.909 | 1.109 |
| 128 | 8 | 1405.57 | 1393.55 | Nano Graph | 1.009 | 1.026 |
| 128 | 16 | 2104.61 | 2093.10 | Nano Graph | 1.005 | 1.013 |
| 512 | 1 | 171.83 | 240.92 | vLLM | 0.713 | 1.075 |
| 512 | 8 | 551.72 | 842.06 | vLLM | 0.655 | 0.948 |
| 512 | 16 | 634.72 | 1036.08 | vLLM | 0.613 | 1.235 |
| 2048 | 1 | 79.05 | 156.41 | vLLM | 0.505 | 1.061 |
| 2048 | 8 | 174.06 | 310.18 | vLLM | 0.561 | 2.002 |
| 2048 | 16 | - | - | incomplete | - | - |

## Trend Summary

- Nano Graph is approximately tied with or slightly faster than vLLM at short context with batch sizes 8 and 16 (`input_len=128`).
- vLLM is faster for the measured longer-context workloads (`input_len=512` and `2048`) and for single-sequence short-context workload.
- The heaviest workload (`input_len=2048,num_seqs=16`) is incomplete because Hugging Face preflight OOMed on this host; it is not reported as a four-way comparison.
- No robust crossover point beyond the short-context batch 8/16 cases is established by this 3x3 matrix.

## Boundary

These results apply to Qwen3-0.6B on this 2 x RTX 2080 Ti host under the listed offline workloads. They should not be generalized to production serving behavior, other models, other GPUs, online latency, or dynamic traffic.
