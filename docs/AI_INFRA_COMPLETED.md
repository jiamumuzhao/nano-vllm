# Nano-vLLM Completed AI Infra Work

This file records capabilities that are implemented in the codebase. `Verified` means there is an automated test, command, or recorded baseline. `Needs more evidence` means the implementation exists but still needs broader tests, integration coverage, or benchmark data.

Last reviewed: 2026-07-12

## Inference execution and model layer

- [x] **Qwen3 inference and weight loading**: `Qwen3ForCausalLM` and safetensors loading are implemented.
- [x] **Tensor Parallel**: worker ranks are initialized with `torch.distributed`/NCCL and coordinated through shared memory and events.
- [x] **CUDA Graph**: eager and CUDA Graph decode paths are supported.
- [x] **KV cache preallocation**: cache block count is computed from available GPU memory, model shape, dtype, and configured memory utilization.

Status: implemented. Multi-GPU throughput and scaling efficiency still need evidence.

## Attention kernels and KV cache

- [x] **Paged KV Cache**: block allocation, release, reference counting, and block tables are implemented.
- [x] **Prefix Caching**: full token blocks are reused through chained block hashes.
- [x] **Triton KV-cache store kernel**: new K/V states are written into physical cache slots.
- [x] **Triton varlen prefill attention**: causal prefill with GQA head mapping.
- [x] **Triton paged prefill attention**: cached prefixes are read directly from page tables.
- [x] **Triton paged decode attention**: decode walks page tables and valid context lengths with online softmax.

Verification: attention correctness tests cover varlen prefill, paged prefill, and paged decode in the current test suite.

## Scheduling and request management

- [x] **Continuous batching scheduler**: waiting and running queues drive prefill and decode.
- [x] **Chunked prefill**: long prompts are split under `max_num_batched_tokens`.
- [x] **Fairer prefill scan**: each waiting request is scanned at most once per scheduling step so long prompts do not monopolize the budget.
- [x] **Preemption and block reclamation**: when decode cannot append KV blocks, requests can be preempted, deallocated, and returned to waiting.
- [x] **Per-token events**: decode postprocessing returns `(seq_id, token_id, finished)` events for streaming consumers.

Verification: scheduler and block-manager tests cover block reuse/release, token-budget behavior, and long/short request scheduling.

## Serving and API

- [x] **AsyncEngine**: async request intake, background stepping, and per-request output queues.
- [x] **Streaming generation**: async iterators yield cumulative text, delta text, token IDs, and finish state.
- [x] **OpenAI-style API**: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, and `/health` are implemented.
- [x] **SSE streaming**: completions and chat completions support `stream: true`.

Status: implemented. Cancellation, timeout, backpressure, readiness, full OpenAI compatibility, and integration tests remain roadmap work.

## Validation and performance records

- [x] **Unit and kernel correctness tests**: recent project baseline reports `/opt/anaconda3/bin/python -m pytest -q` as `7 passed`.
- [x] **Transformers alignment script**: `scripts/verify_against_transformers.py` exists; historical record shows sampled prefill logits passed with `max_abs_error=0.066406` and top-1 match.
- [x] **Single-workload generation throughput record**: historical record shows about `1410 tok/s` for CUDA Graph and about `306 tok/s` for eager mode on Qwen3-0.6B with 8 concurrent requests and a 128/64 token workload.
- [x] **Unified offline benchmark tooling**: `scripts/benchmark_offline.py`, `scripts/benchmark_offline_worker.py`, and `scripts/benchmark_schema.py` run selected offline engines with shared workload/schema and emit JSONL plus Markdown.
- [x] **Same-condition offline baseline**: [docs/benchmarks/offline_comparison_strict_2026-07-12.md](benchmarks/offline_comparison_strict_2026-07-12.md) compares Nano-vLLM eager, Nano-vLLM CUDA Graph, Hugging Face `generate()`, and vLLM on identical prompt token IDs, fixed output length, aligned max sequence/batched-token constraints, and 10 measured runs.
- [x] **vLLM isolated feasibility**: [docs/benchmarks/vllm_feasibility_2026-07-12.md](benchmarks/vllm_feasibility_2026-07-12.md) records the project-local `.venv-vllm-bench` runner created with `--system-site-packages`, the reused preinstalled `vllm==0.24.0` package, and successful smoke/formal vLLM measurement.
- [x] **3x3 offline workload matrix**: [docs/benchmarks/offline_matrix_2026-07-12.md](benchmarks/offline_matrix_2026-07-12.md) compares Nano-vLLM eager, Nano-vLLM CUDA Graph, Hugging Face `generate()`, and vLLM across `num_seqs=1,8,16` and `input_len=128,512,2048` with fixed output length, shared prompt token IDs, shared measured seeds, `max_model_len=4096`, and `max_num_batched_tokens=32768`. Eight workloads completed formal four-engine measurement; `input_len=2048,num_seqs=16` is recorded as incomplete because Hugging Face preflight OOMed.
- [x] **Long-context Nano-vLLM profiling**: [docs/profiling_long_context.md](profiling_long_context.md) and [docs/profiles/long_context_profile_2026-07-12.md](profiles/long_context_profile_2026-07-12.md) record CPU/CUDA profiler evidence for short vs long context in eager and CUDA Graph modes. The phase explains bottlenecks but does not implement optimizations.

Status: default offline baseline evidence, a 3x3 offline matrix, Nano-vLLM long-context profiler evidence, and a real concurrent AsyncEngine serving benchmark exist. Online HTTP/network load testing, larger workload matrices, performance regression gates, and optimized long-context kernels are not complete.

## Documentation phase completed

- [x] **README narrative alignment**: [README.md](../README.md) now positions the project as an educational/lightweight inference engine and removes unsupported vLLM parity claims.
- [x] **Architecture document**: [docs/architecture.md](architecture.md) documents request flow, scheduler behavior, KV cache design, attention paths, execution, serving, and limits.
- [x] **Benchmark evidence document**: [docs/benchmark.md](benchmark.md) separates measured results from unmeasured gaps and defines a future benchmark protocol.
- [x] **Roadmap alignment**: [OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md) now marks paged attention, AsyncEngine, OpenAI-style API, and SSE as completed while leaving quantization, observability, strict comparisons, and multi-model support as future work.

The second phase added real single-workload offline benchmark measurements. The third phase added a 3x3 offline workload matrix, with one heaviest workload recorded as incomplete due to Hugging Face OOM. The fourth phase added Nano-vLLM long-context profiling and bottleneck analysis. The serving phase added `scripts/benchmark_serving.py` with concurrent AsyncEngine streams, request-level TTFT/TPOT/E2E measurements, service metrics snapshots, KV preflight diagnostics, and JSONL/Markdown output. These phases did not add HTTP/network load testing or kernel optimizations.

## Tracking rule

When new work lands, update:

1. This file with implemented capability, implementation location, verification status, and limitations.
2. [docs/AI_INFRA_OPTIMIZATION_PLAN.md](AI_INFRA_OPTIMIZATION_PLAN.md) with plan status and links to evidence.
3. README and docs only with claims supported by code, tests, or recorded benchmark artifacts.
