# Benchmark Evidence and Protocol

This document separates historical Nano-vLLM-only measurements, offline generation benchmarks,
and the direct online AsyncEngine serving benchmark. Online smoke artifacts are evidence of the
measurement protocol, not a completed large-matrix performance claim.

## Benchmark tools

The reproducible offline benchmark entry points are:

- `scripts/benchmark_schema.py`: shared environment, prompt-token, JSONL, and summary helpers.
- `scripts/benchmark_offline_worker.py`: one-engine worker process for `nano_eager`, `nano_graph`, `hf`, or `vllm`.
- `scripts/benchmark_offline.py`: driver that runs selected engines, records run-level JSONL records, records summary records, and writes Markdown.
- `scripts/benchmark_offline_matrix.py`: workload-matrix driver that reuses the strict offline benchmark protocol across `(input_len, num_seqs)` combinations.
- `scripts/check_vllm_feasibility.py`: read-only host/vLLM feasibility audit.

Strict workload-compatible command:

```bash
.venv-vllm-bench/bin/python scripts/benchmark_offline.py \
  --engines nano_eager,nano_graph,hf,vllm \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 8 \
  --input-len 128 \
  --output-len 64 \
  --warmup-runs 3 \
  --runs 10 \
  --dtype float16 \
  --seed 1234 \
  --temperature 1.0 \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.5 \
  --output-jsonl docs/benchmarks/offline_comparison_strict_2026-07-12.jsonl \
  --output-md docs/benchmarks/offline_comparison_strict_2026-07-12.md
```

## Environment

Source: [docs/baseline.md](baseline.md) and the run-level JSONL records.

- Host alias: `Ubuntu`
- Project path: `/root/workspace/nano-vllm`
- Python: `/opt/anaconda3/bin/python` 3.13.9
- PyTorch: 2.11.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Triton: 3.6.0
- Transformers: 5.12.1
- GPUs: 2 x NVIDIA GeForce RTX 2080 Ti, compute capability 7.5
- Model: `/root/huggingface/Qwen3-0.6B`
- Git revision during unified benchmark: `5fc5453e1245fbd31f3906ca6a618f8276690c1b` with a dirty worktree containing documentation and benchmark-script changes.

## Correctness commands

Unit and kernel tests:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python -m pytest -q
```

Recent project baseline before this phase: `7 passed`.

Historical Hugging Face alignment command:

```bash
cd /root/workspace/nano-vllm
/opt/anaconda3/bin/python scripts/verify_against_transformers.py \
  --model /root/huggingface/Qwen3-0.6B \
  --prompt-len 16
```

Historical result from [docs/baseline.md](baseline.md): sampled prefill logits passed with `max_abs_error=0.066406` and matching top-1 token under the configured threshold.

## Historical Nano-vLLM-only baseline

The historical workload measured offline `llm.generate()` throughput after warmup:

- Tensor parallel size: 1
- Sequences: 8
- Prompt length: 128
- Requested output length: 64
- Max model length: 512
- Max sequences: 8
- GPU memory utilization: 0.5
- Warmup runs: 1
- Measured runs: 3

| System/mode | Measured metric | Result |
| --- | --- | ---: |
| Nano-vLLM eager | End-to-end output tokens per second | 305.78361946551905 |
| Nano-vLLM CUDA Graph | End-to-end output tokens per second | 1410.4785356279172 |

These numbers compare Nano-vLLM eager vs Nano-vLLM CUDA Graph for one historical workload. They are not a vLLM or Hugging Face comparison.

## Strict controlled offline benchmark, 2026-07-12

Raw evidence:

- Strict report: [benchmarks/offline_comparison_strict_2026-07-12.md](benchmarks/offline_comparison_strict_2026-07-12.md)
- Strict JSONL: [benchmarks/offline_comparison_strict_2026-07-12.jsonl](benchmarks/offline_comparison_strict_2026-07-12.jsonl)
- vLLM feasibility record: [benchmarks/vllm_feasibility_2026-07-12.md](benchmarks/vllm_feasibility_2026-07-12.md)

Protocol:

- Same model path: `/root/huggingface/Qwen3-0.6B`.
- Same deterministic prompt token IDs, generated from the same tokenizer and seed; prompt token SHA-256: `a412c076e2a785e01d46fb9b53c33868832776b9930fc50d7b5ec325d8cc9de1`.
- Same workload: 8 sequences, 128 input tokens per sequence, 64 output tokens per sequence, warmup=3, measured runs=10.
- Same batch constraints: tensor parallel size `1`, max model length `512`, max sequences `8`, max batched tokens `1024`, GPU memory utilization `0.5`.
- Same dtype: `float16`; same temperature: `1.0`; stochastic sampling, not greedy.
- Fixed output length: Nano-vLLM uses `ignore_eos=True`; Hugging Face uses `min_new_tokens=max_new_tokens` and `eos_token_id=None`; vLLM uses `ignore_eos=True` plus `min_tokens=max_tokens`.
- Measured generation seeds are explicitly recorded and shared across engines: `11234` through `11243`. vLLM uses `SamplingParams(seed=...)` for each warmup and measured run.
- Timing excludes model/tokenizer initialization and warmup, and synchronizes CUDA around each measured generation run.

Results:

| Engine | Mode | Mean output tok/s | Stddev | Min | Max | Run seconds |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Nano-vLLM | eager | 311.09 | 1.66 | 308.17 | 312.97 | 1.6370, 1.6434, 1.6448, 1.6603, 1.6419, 1.6451, 1.6396, 1.6495, 1.6614, 1.6360 |
| Nano-vLLM | CUDA Graph | 1412.67 | 2.27 | 1409.66 | 1417.26 | 0.3613, 0.3627, 0.3628, 0.3632, 0.3631, 0.3625, 0.3621, 0.3625, 0.3621, 0.3620 |
| Hugging Face | generate | 302.23 | 0.25 | 301.92 | 302.55 | 1.6924, 1.6936, 1.6956, 1.6934, 1.6923, 1.6948, 1.6926, 1.6956, 1.6949, 1.6958 |
| vLLM | offline generate | 1394.37 | 15.62 | 1376.34 | 1413.27 | 0.3720, 0.3711, 0.3693, 0.3696, 0.3623, 0.3623, 0.3624, 0.3695, 0.3705, 0.3632 |

vLLM status: measured. The vLLM row was run by a project-local venv runner created with `--system-site-packages`, reusing the preinstalled `vllm==0.24.0` package from the Conda environment. This avoids adding vLLM to project dependencies, but it is not a fully independent dependency environment.

## 3x3 offline workload matrix, 2026-07-12

Raw evidence:

- Matrix report: [benchmarks/offline_matrix_2026-07-12.md](benchmarks/offline_matrix_2026-07-12.md)
- Matrix JSONL: [benchmarks/offline_matrix_2026-07-12.jsonl](benchmarks/offline_matrix_2026-07-12.jsonl)
- Matrix verification JSON: [benchmarks/offline_matrix_2026-07-12.verify.json](benchmarks/offline_matrix_2026-07-12.verify.json)
- Per-workload evidence: [benchmarks/matrix/](benchmarks/matrix/)

Command:

```bash
.venv-vllm-bench/bin/python scripts/benchmark_offline_matrix.py \
  --engines nano_eager,nano_graph,hf,vllm \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs-list 1,8,16 \
  --input-lens 128,512,2048 \
  --output-len 64 \
  --warmup-runs 2 \
  --runs 5 \
  --dtype float16 \
  --seed 1234 \
  --temperature 1.0 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.5 \
  --matrix-dir docs/benchmarks/matrix \
  --output-jsonl docs/benchmarks/offline_matrix_2026-07-12.jsonl \
  --output-md docs/benchmarks/offline_matrix_2026-07-12.md
```

Protocol:

- Same model, dtype, tensor parallel size, fixed-output policy, deterministic prompt token IDs, temperature, and per-run measured seeds as the strict benchmark.
- Matrix workload: `num_seqs=1,8,16` x `input_len=128,512,2048`, output length `64`, warmup runs `2`, measured runs `5`.
- Same batch/token constraints where a workload is measured: `max_model_len=4096`, `max_num_batched_tokens=32768`, and `max_num_seqs` equal to the workload `num_seqs`.
- Preflight is recorded for every workload before formal measurement. The `input_len=2048,num_seqs=16` workload is incomplete because Hugging Face preflight OOMed on this 10.56 GiB GPU host, so it is not reported as a four-way formal comparison.

Summary:

| input_len | num_seqs | Nano eager tok/s | Nano Graph tok/s | HF tok/s | vLLM tok/s | Nano Graph / vLLM | Nano eager / HF | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 1 | 42.44 | 251.98 | 38.27 | 277.07 | 0.909 | 1.109 | complete |
| 128 | 8 | 298.60 | 1405.57 | 291.08 | 1393.55 | 1.009 | 1.026 | complete |
| 128 | 16 | 571.12 | 2104.61 | 563.86 | 2093.10 | 1.005 | 1.013 | complete |
| 512 | 1 | 40.08 | 171.83 | 37.30 | 240.92 | 0.713 | 1.075 | complete |
| 512 | 8 | 233.69 | 551.72 | 246.43 | 842.06 | 0.655 | 0.948 | complete |
| 512 | 16 | 368.87 | 634.72 | 298.79 | 1036.08 | 0.613 | 1.235 | complete |
| 2048 | 1 | 35.44 | 79.05 | 33.42 | 156.41 | 0.505 | 1.061 | complete |
| 2048 | 8 | 136.25 | 174.06 | 68.07 | 310.18 | 0.561 | 2.002 | complete |
| 2048 | 16 | - | - | - | - | - | - | incomplete: HF preflight OOM |

Nano-vLLM CUDA Graph is roughly tied with or slightly faster than vLLM at `input_len=128,num_seqs=8/16`; vLLM is faster for the measured longer-context workloads and for the single-sequence short-context workload. This is a 3x3 offline matrix for this model/GPU/protocol only, not a production-serving benchmark.

## Long-context profiling, 2026-07-12

Raw evidence:

- Profiling overview: [profiling_long_context.md](profiling_long_context.md)
- Detailed profile report: [profiles/long_context_profile_2026-07-12.md](profiles/long_context_profile_2026-07-12.md)
- Machine-readable summary: [profiles/long_context_profile_2026-07-12.json](profiles/long_context_profile_2026-07-12.json)

The profiling phase measures Nano-vLLM only, using the strict matrix's `num_seqs=8`, `output_len=64`, `max_model_len=4096`, `max_num_batched_tokens=32768`, and same-prompt warmup behavior. It does not profile vLLM internals.

Key finding: for Nano-vLLM CUDA Graph, moving from `input_len=128` to `input_len=2048` increases mean synchronized decode-step latency from about `4.61 ms` to about `10.08 ms`. The long-context CUDA profile is dominated by `_paged_prefill_kernel` at about `2191.89 ms`, with `_decode_paged_kernel` at about `360.21 ms` across the captured steps. This supports prioritizing split-KV / partitioned-softmax paged decode and paged-prefill/block-table analysis next; no optimization has been implemented yet.

## Not yet measured

- Online serving throughput and latency.
- TTFT, TPOT, P50/P95/P99 latency.
- Peak memory usage and KV cache utilization under load.
- Prefix-cache hit rate and impact on shared-prefix workloads.
- Tensor parallel scaling efficiency.
- Larger workload matrices across higher concurrency, additional prompt lengths, additional output lengths, larger model lengths, and shared-prefix workloads.

## Known gaps

- The unified comparison is one offline workload on one host. It does not establish broad performance superiority or production readiness.
- The engines use different internal implementations and batching policies; the protocol controls inputs and generation settings but cannot make implementation details identical.
- Sampling is stochastic. The benchmark fixes seeds and sampling parameters for reproducibility, but it is not a greedy or exact-output-equivalence benchmark.

## Proposed next benchmark protocol

Future performance claims should extend this fixed protocol:

1. Record environment: git commit, GPU model/count, driver, CUDA, PyTorch, Triton, Transformers, Python, model path, dtype, tensor parallel size, and engine flags.
2. Fix workload matrix: concurrency `1/8/32/128`, input length `128/1024/4096/8192`, output length `32/128/512`, and representative shared-prefix workloads.
3. Use warmup runs before measurement and at least three measured repetitions per workload.
4. Capture metrics: end-to-end throughput, prefill throughput, decode throughput, TTFT, TPOT, P50/P95/P99 latency, peak memory, KV block usage, prefix-cache hit rate, and preemption count.
5. Compare fairly: use the same model weights, dtype, sampling settings, max model length, batch limits, prompt set, output length policy, and hardware for Nano-vLLM eager, Nano-vLLM CUDA Graph, Hugging Face, and vLLM.
6. Store raw results in machine-readable form before publishing summary tables.
7. Publish only supported claims and keep offline throughput separate from online serving metrics.


## Paged prefill Q-tile selector evidence

The Q-tile measurements are separated into two evidence classes:

- Microbenchmark: scripts/benchmark_paged_prefill_q_tile.py measures the attention kernel with parameter sweeps and preserves every serial/parallel result, including regressions. Historical microbenchmark files are not selector evidence.
- Full-model end-to-end: scripts/benchmark_qwen3_eager_q_tile.py measures Qwen3-0.6B eager generation with identical prompt tokens, seed, sampling parameters, warmups, and repeats for serial and parallel modes.

The formal selector evidence is the post-fix matrix:

- JSON: docs/benchmarks/qwen3_eager_q_tile_20260720T032305Z.json
- Markdown: docs/benchmarks/qwen3_eager_q_tile_20260720T032305Z.md
- Protocol: float16, TP=1, enforce eager, CUDA Graph disabled, batch 1/4/8, input 1024/2048/4096, output 64, warmup 3, repeats 10, RTX 2080 Ti.

The selector is derived only from that full-model matrix. Because the observed end-to-end differences were small, the latest formal matrix showed no clear parallel benefit, so auto safely uses serial for all workloads. The selector still computes batch_size, max_seqlen_q, and ceil(max_seqlen_q / BLOCK_Q), with configurable thresholds reserved for a future revalidated matrix. Explicit serial and parallel modes remain available, and legacy paged_prefill_q_tile_parallel=True/False remains a forced-path compatibility interface.


## W8A16 weight-only benchmark

scripts/benchmark_w8a16.py compares the existing none FP16 path with w8a16 in eager mode and writes JSON plus Markdown under docs/benchmarks/. The benchmark reports load-time allocated memory, peak memory, prefill latency/throughput, decode step latency/throughput, end-to-end throughput, and raw runs. It does not assume W8A16 is faster.

The first implementation uses per-output-channel symmetric weight-only quantization. Activations, KV cache, RMSNorm, embeddings, and LM head remain FP16. Quantization occurs once after normal TP-aware weight loading and before warmup. W8A16 uses a runtime policy that forces eager execution and skips CUDA Graph capture/replay; FP16 quantization=none retains the original CUDA Graph behavior. W8A16 does not silently fall back to FP16 weights.


## W8A16 TP=2 validation records

`scripts/record_w8a16_tp2_validation.py` runs the real two-GPU NCCL correctness test and records the subprocess command, exit code, status, duration, GPU/PyTorch/CUDA/Triton/NCCL environment, Git state, pytest stdout/stderr, and covered TP semantics. It is not a throughput benchmark.

Run:

```bash
/opt/anaconda3/bin/python scripts/record_w8a16_tp2_validation.py
```

The script forces `NANOVLLM_RUN_TP2=1 CUDA_VISIBLE_DEVICES=0,1`, writes atomically to `docs/benchmarks/w8a16_tp2_validation_<timestamp>.json` and `.md`, and preserves failed or timed-out diagnostics. The current formal record is `docs/benchmarks/w8a16_tp2_validation_20260720T082005Z.json` and `.md`.


## W8A16 TP=2 end-to-end inference benchmark

`scripts/benchmark_w8a16_tp2_e2e.py` runs real `LLMEngine(..., tensor_parallel_size=2)` requests for FP16 (`quantization=none`) and W8A16, with independent engine lifetimes per workload/mode. The formal matrix covers batch `1,4`, prompt lengths `128,1024`, output length `32`, warmup `3`, and repeats `10`; prompts, seeds, sampling, and scheduler limits are identical between modes.

Formal command:

```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/anaconda3/bin/python scripts/benchmark_w8a16_tp2_e2e.py \
  --model /root/huggingface/Qwen3-0.6B --batch-sizes 1,4 \
  --input-lengths 128,1024 --output-len 32 --warmup-runs 3 \
  --repeats 10 --enforce-eager
```

The `--enforce-eager` flag is the fair kernel comparison: both modes are eager. W8A16 is always eager-only; FP16 may retain CUDA Graph under the default policy, so a default-policy run is a serving-policy comparison rather than an isolated quantization comparison. Results include TTFT, prefill/decode latency and throughput, end-to-end throughput, memory, graph policy, and raw repeats. The formal record is `docs/benchmarks/w8a16_tp2_e2e_20260720T084653Z.json` and `.md`.


## W8A16 codegen evidence gate

## W8A16 CUDA WMMA extension

## W8A8 experimental benchmark-only path

`scripts/benchmark_w8a8_decode.py` measures activation quantization, INT8 GEMM, and end-to-end W8A8 Linear separately against FP16 `F.linear` and W8A16 legacy. The record is `docs/benchmarks/w8a8_decode_20260722T080255Z.json`/`.md`, with six decode workloads, 30 warmups, 100 iterations, and IMMA evidence. W8A8 is not connected to ModelRunner or model benchmarks. Its integration gate is false because M=1/4 end-to-end W8A8 is still more than 10% slower than FP16 `F.linear`, despite being more than 10% faster than W8A16 legacy.

The CTA-inclusive record is `docs/benchmarks/w8a8_decode_20260722T082303Z.json`/`.md`. It covers `M=1,4,16,32,64`, `N=1024,4096`, `K=1024`, and reports activation quantization, legacy/CTA INT8 GEMM, and end-to-end latency with raw samples. Runtime selection is legacy for `M=1/4` and CTA for `M>=16` when `N,K>=1024`. The corresponding codegen evidence is `docs/benchmarks/w8a8_tensorcore_codegen_20260722T082225Z.json`, with `IMMA.8816.S8.S8` for both representative paths. The future integration gate is false; W8A8 remains isolated and no model integration is implied.

The current split-candidate evidence is `docs/benchmarks/w8a8_decode_20260722T092147Z.json`/`.md` with codegen `docs/benchmarks/w8a8_tensorcore_codegen_20260722T092029Z.json`. It measures FP16, W8A16 legacy, W8A8 legacy, CTA16, and CTA32, including activation quantization, GEMM, and end-to-end stages. CTA32 is faster than W8A8 legacy on selected large shapes, but no shape satisfies both the required W8A16 improvement and FP16 slowdown limits. Consequently `auto` has an empty exact-shape whitelist and selects legacy for every workload.

## W8A8 operator-level break-even

`scripts/benchmark_w8a8_break_even.py` measures only the Linear operator path: activation quantization, INT8 GEMM, scale/bias epilogue, and output writeback. It does not represent full-model prefill because W8A8 is not connected to `ModelRunner`. The formal record is `docs/benchmarks/w8a8_break_even_20260723T022734Z.json`/`.md`, with `M=1,4,16,32,64,128,256,512,1024,2048`, `N=1024/4096`, `K=1024`, warmup 30, and 100 measured iterations. No tested M reached the joint W8A16 +10% speedup and FP16 <=10% slowdown gate for any implementation or either N.

## W8A8 cuBLASLt experimental baseline

The cuBLASLt path is `w8a8_experimental_cublaslt_int8_i32`. It uses one-time int8 weight packing, cached cuBLASLt INT8×INT8→INT32 plans, preallocated `W8A8Workspace`, and a separate INT32-to-FP16 scale/bias epilogue. API evidence is `docs/benchmarks/w8a8_cublaslt_api_20260723T025452Z.json`; it is explicitly `cublaslt_api_verified`, not a custom extension SASS claim, and records rejection of mismatched or in-place-mutated weights. The latest comparison is `docs/benchmarks/w8a8_break_even_20260723T025525Z.json`/`.md`, covering M `1..2048`, N `1024/4096`, with quantization, GEMM, epilogue, and operator end-to-end timings. No shape satisfied both the W8A16 and FP16 integration thresholds. W8A8 remains benchmark-only and has no model-e2e prefill result.

The latest cuBLASLt autotune comparison is
`docs/benchmarks/w8a8_break_even_20260723T082812Z.json`/`.md`, with API/layout
evidence in `docs/benchmarks/w8a8_cublaslt_api_20260723T082738Z.json`. It keeps
an explicit row-major baseline beside the autotuned row-major path, records
all budget candidates and skip reasons, and verifies stable workspace reuse.
All budgets selected the same `heuristic_index_0` on this RTX 2080 Ti; COL32
is an explicit structured skip because its transform/packed-buffer path is not
validated. This offline evidence does not alter W8A8 benchmark-only routing.

The latest COL32-aware API record is
`docs/benchmarks/w8a8_cublaslt_api_20260723T092526Z.json`. Both representative
shapes completed the actual row-major-to-COL32 transform, but the subsequent
cuBLASLt heuristic returned zero valid candidates with status code 0
(`CUBLAS_STATUS_SUCCESS`), so no COL32 matmul was run. The selected layout
remained row-major; the benchmark record is
`docs/benchmarks/w8a8_break_even_20260723T092600Z.json`/`.md`.

## Quantization production routing decision

The offline record generated by `scripts/record_quantization_routing_decision.py` fixes the production choices: `quantization=none` is the default FP16 performance route, and explicit `quantization=w8a16` is the weight-memory optimization route. W8A16 is not automatically selected and is not presented as a guaranteed decode-speed path on the RTX 2080 Ti. W8A8 WMMA and cuBLASLt remain benchmark-only; Config rejects W8A8 variants before model initialization with a clear error. Runtime routing does not read benchmark JSON. Reconsideration requires a reproducible workload matrix passing the existing gate and subsequent real model end-to-end validation.

The CTA tiling benchmark `scripts/benchmark_w8a16_wmma_tiling.py` compares the retained legacy single-warp kernel, optimized CTA WMMA, and FP16 `F.linear` for decode `M=1,4,16; N=1024,4096` and prefill `M=64,256; N=1024,4096`, all with `K=1024`, 10 warmups and 30 timed iterations. The current record is `docs/benchmarks/w8a16_wmma_tiling_20260722T025201Z.json`/`.md`; results are reported without filtering regressions.

The final production selector is conservative: decode `M<=16` uses legacy WMMA; prefill CTA is selected only for `M>=256,N>=1024,K>=1024`, while other prefill uses legacy WMMA. The selector-aware record is `docs/benchmarks/w8a16_wmma_tiling_20260722T031653Z.json`/`.md` and includes `selected_runtime_path` for every raw candidate measurement. The policy is RTX 2080 Ti evidence, not a universal GPU performance claim.

The decode-wide candidate gate record is `docs/benchmarks/w8a16_wmma_tiling_20260722T072235Z.json`/`.md`. It uses warmup `30` and `100` iterations over `M=1,4,16`, `N=1024,4096`, `K=1024`, and retains legacy, candidate, and FP16 `F.linear` raw measurements. Candidate P50 speedups versus legacy were all negative (−34.78% to −44.91%), so the required +5% all-workload gate failed and no TP=1/TP=2 production benchmark was generated for this candidate.

The prebuilt extension is built with `CUDA_HOME=/usr/local/cuda /opt/anaconda3/bin/python setup.py build_ext --inplace`; it is never JIT-compiled in the inference hot path. `scripts/inspect_w8a16_tensorcore_codegen.py` runs the representative `M=32,N=32,K=1024` extension kernel and persists PTX/SASS evidence. The accepted record is `docs/benchmarks/w8a16_tensorcore_codegen_20260722T020853Z.json`, whose SASS contains `HMMA.1688.F32` and whose runtime path is `cuda_fp16_tensorcore_wmma`. Benchmark metadata reads this record and does not claim Tensor Core execution when the extension or evidence is unavailable. W8A16 remains eager-only; BF16, CPU, unsupported devices and extension failures use explicit fallbacks.

`scripts/inspect_w8a16_codegen.py` compiles a representative W8A16 kernel in a fresh `TRITON_CACHE_DIR`, persists the discovered PTX/cubin evidence next to `docs/benchmarks/w8a16_codegen_<timestamp>.json`, and records whether PTX contains FP16 MMA or only `fma.rn.f32`. A performance record is permitted to claim `fp16_mma_verified` only when PTX contains an FP16 MMA signature or available SASS contains `HMMA` (including on SM75). The current RTX 2080 Ti record is `fp32_fma_fallback`, so no new FP16-MMA performance result was generated.

## Offline performance regression gate

`scripts/check_benchmark_regression.py` is a CPU-only comparison tool. It reads existing benchmark JSONL records and compares only `record_type="summary"`, `status="ok"` records with matching workload identity. The identity fields are:

`engine`, `model`, `dtype`, `tensor_parallel_size`, `num_seqs`, `input_len`, `output_len`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, and `temperature`.

The default gate requires at least 3 measured runs per side, uses median output tok/s and median generation seconds, fails for throughput regressions above 10% or latency regressions above 15%, and requires matching GPU name/count, CUDA runtime, and PyTorch version. These are initial offline protection thresholds, not formal CI SLOs. `--allow-environment-mismatch` explicitly permits a non-strict comparison and records that limitation in JSON/Markdown.

Every measured run must contain finite, strictly positive `output_tokens_per_second` and `seconds` values. Missing fields, non-dict measurements, non-convertible values, `NaN`, infinities, zero, and negative values fail the gate with the workload identity and run index; invalid runs are never dropped before median calculation.

Reproducible workflow:

```bash
/opt/anaconda3/bin/python scripts/benchmark_offline.py \
  --engines nano_eager --runs 3 --output-jsonl docs/benchmarks/baseline.jsonl \
  --output-md docs/benchmarks/baseline.md

# Run the candidate from the changed checkout, with the same workload arguments.
/opt/anaconda3/bin/python scripts/benchmark_offline.py \
  --engines nano_eager --runs 3 --output-jsonl docs/benchmarks/candidate.jsonl \
  --output-md docs/benchmarks/candidate.md

/opt/anaconda3/bin/python scripts/check_benchmark_regression.py \
  docs/benchmarks/baseline.jsonl docs/benchmarks/candidate.jsonl \
  --output-dir docs/benchmarks
```

The gate does not run a model or call GPU APIs. Online serving indicators such as TTFT, TPOT, P99 latency, KV usage, cache hit rate, and preemption remain 1.1 work and are not implied by this offline result.
## Online AsyncEngine serving benchmark

`scripts/benchmark_serving.py` drives real concurrent `AsyncEngine.generate()` streams. It is
separate from the offline `LLM.generate()` benchmarks above: each measured run creates and
destroys an independent AsyncEngine, submits deterministic token-id prompts concurrently, and
records request-level output timestamps and finish reasons.

Example smoke run:

```bash
/opt/anaconda3/bin/python scripts/benchmark_serving.py \
  --model /root/huggingface/Qwen3-0.6B \
  --concurrencies 1 --input-lens 32 --output-lens 4 \
  --warmup-runs 1 --runs 1 --max-model-len 128 \
  --max-num-seqs 1 --max-num-batched-tokens 64 \
  --gpu-memory-utilization 0.9 \
  --output-jsonl docs/benchmarks/serving_smoke.jsonl \
  --output-md docs/benchmarks/serving_smoke.md
```

`0.9` is the successfully observed smoke configuration for the current RTX 2080 Ti /
Qwen3-0.6B environment. It is only for validating the serving-metrics pipeline, not a
production recommendation or a cross-GPU minimum.

The default CLI matrix is intentionally large (`1,8,32,128` concurrency × `128,1024,4096,8192`
input × `32,128,512` output) and is not run automatically by acceptance tests. A complete
matrix requires real GPU time and its JSONL/Markdown artifacts must be generated before making
performance claims.

Metric definitions are explicit: TTFT is request submission to first output; TPOT is the mean
interval between adjacent output tokens for one request and is `null` for a one-token
completion; global output tok/s is all completed output tokens divided by
`max(request.completed_at_absolute) - min(request.submitted_at_absolute)`. Engine creation,
barrier creation, task creation, and batch-start timestamps are not substitutes for the first
actual request submission. If timestamps are missing, invalid, or reversed, global seconds
and throughput are `null` with a failure reason. Prefill/decode tok/s remain `null` when the
current AsyncEngine cannot expose an unambiguous phase boundary. P50/P95/P99 use sorted linear
interpolation at `(n-1)*p/100`, with empty samples represented as `null`. Failed, cancelled,
timeout, queue-full, or engine-fatal workloads never receive fabricated throughput values.
Each record includes Scheduler/AsyncEngine snapshots for KV current/peak usage, prefix-cache
hit/token-hit rates, preemption, and request terminal counts.

Prompt fingerprints are SHA-256 over canonical compact JSON of the final complete prompt-token
list after any `prefix-sharing-ratio` substitution. Repeating the same seed and arguments gives
the same SHA; changing the sharing ratio or submitted tokens changes it.

Before constructing `AsyncEngine`, the benchmark performs a read-only CUDA/model-config KV
preflight. It records CUDA free/total bytes, target bytes, estimated model bytes, one-block
bytes, configured capacity estimate, TP, dtype, block size, and a theoretical minimum
utilization for one KV block.
The preflight does not initialize NCCL or allocate model/KV tensors. The verified RTX 2080 Ti /
Qwen3-0.6B smoke command uses `--gpu-memory-utilization 0.9` with the small workload above;
this is only a smoke configuration, not a performance conclusion. The field
`theoretical_minimum_utilization_for_one_kv_block` is only the model-estimate plus one-block
lower bound; it is not a suggested benchmark setting. The conservative recommendation is to
increase utilization or reduce `max_model_len`/`max_num_seqs`, while recognizing that warmup
workspace and allocator fragmentation can still cause runtime failure. In the current observed
environment, `0.25` reached runtime KV allocation failure and `0.9` passed the single-request
smoke; neither is a cross-GPU minimum. The `0.25` result is retained only as preflight/runtime
diagnostic evidence and is expected not to produce valid performance metrics. The field
`theoretical_minimum_utilization_for_one_kv_block` is a theoretical lower bound, not a usable
recommendation or guarantee of successful model loading, warmup, allocator behavior, or KV
allocation. The conservative action is to increase utilization or reduce
`max_model_len`/`max_num_seqs` and then verify the real smoke run.

## Model correctness / logits alignment

`scripts/verify_logits_against_transformers.py` is a correctness verifier, not a performance
benchmark. It compares the full vocabulary logits from Nano-vLLM with the same local
Qwen3-0.6B weights loaded by Transformers, using FP16, CUDA eager execution, TP=1, and the
same deterministic prompt and continuation token ids. It covers prefill lengths `1, 16, 17,
31, 32, 128, 1024`, a mixed batch `[17, 32, 127]`, and eight decode steps. Results include
element count, max/mean absolute error, RMSE, and vocabulary top-1 agreement; padding is not
included in the comparison. The default gate is max absolute error `<= 5e-2`, mean absolute
error `<= 5e-3`, and top-1 agreement `== 1.0`.

Run it only when the local model and CUDA are available:

```bash
/opt/anaconda3/bin/python scripts/verify_logits_against_transformers.py \
  --model /root/huggingface/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --output-dir docs/benchmarks
```

The verifier writes `logits_alignment_qwen3_tp1.jsonl` and
`logits_alignment_qwen3_tp1.md`. This model validates the Qwen3 GQA path; MHA remains covered
by synthetic attention-correctness tests. The validation-only logits hook bypasses sampling
without changing the production serving path or retaining a persistent full-logits workspace.

The latest TP=1 run is recorded in those artifacts and did not pass the strict gate: the
worst max absolute error was `0.1152344`, worst mean absolute error was `0.0094781`, and the
minimum top-1 agreement was `0.875`. The v3 layerwise record
`logits_alignment_qwen3_tp1_layerwise_v3.jsonl` performs a full-model execution-order bisect
with complete Nano/Transformers shapes, dtypes, layouts and valid-token-mask metadata. The
first formal case-0 prefill divergence is `layer_1.k_norm_output`, immediately after
`layer_1.q_norm_output`: max `0.125`, mean `0.00068061`, RMSE `0.00416488`, top-1 `1.0`,
with maximum error at row `0`, KV head `2`, head-dim `52` (`124.5625` vs `124.4375`). Its
input shape is `[1,1,8,128]` on both sides, normalization is only over `dim=-1` (`head_dim`),
and the input discrepancy is max `0.00048828125` (mean `4.34386e-05`, RMSE `7.16107e-05`).
The per-head summary is stored in the v3 JSONL, proving the statistic does not cross heads.
The prior layer-0
RoPE/RMSNorm discrepancies were reduced by matching the Transformers RMSNorm cast/weight order
and RoPE `x*cos + rotate_half(x)*sin` expression. Production FP16 retains the merged gate/up
path; split-gates is validation-only. These artifacts remain a correctness blocker, not a
reason to loosen thresholds.
