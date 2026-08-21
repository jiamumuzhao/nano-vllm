# Nano-vLLM Optimization Roadmap

This roadmap tracks work that would make Nano-vLLM a stronger AI-infra portfolio project. It separates implemented capabilities from future work so the project narrative stays aligned with the current code.
Last reviewed: 2026-08-21

## Completed in the current codebase

The following items are implemented and tracked in [docs/AI_INFRA_COMPLETED.md](docs/AI_INFRA_COMPLETED.md):

- Paged KV cache with physical blocks, per-sequence block tables, reference counts, allocation, reuse, and deallocation.
- Prefix caching through chained hashes of complete token blocks.
- Chunked prefill with a scheduler token budget and waiting/running queues.
- Decode preemption when KV blocks are unavailable.
- Triton KV-cache store kernel.
- Triton varlen prefill attention.
- Triton paged prefill attention for cached prefixes.
- Triton paged decode attention that reads directly from page tables and valid context lengths.
- GQA-aware attention head mapping and online softmax in the custom kernels.
- CUDA Graph decode replay when eager mode is disabled.
- Tensor parallel execution through NCCL worker ranks.
- `AsyncEngine`, per-token output events, OpenAI-style completions/chat-completions endpoints, and SSE streaming.
- Bounded serving lifecycle with request cancellation, timeout handling, queue admission limits, stream backpressure, request status inspection, and terminal error propagation.
- Read-only scheduler and serving metrics snapshots for KV usage, prefix-cache hits, preemption, request states, and queue limits.
- Split-KV decode support, paged-prefill Q-tile selection, W8A16 routing, and benchmark/codegen evidence for the implemented experimental paths.
- Documentation alignment phase: README, architecture document, benchmark evidence document, and roadmap status cleanup.

## P0: Evidence and correctness

- [x] Build a repeatable offline benchmark protocol that records raw JSONL results, environment metadata, git revision, workload, and summary statistics. Online serving protocol remains future work.
- [x] Run same-condition offline comparisons against Hugging Face `generate()` and vLLM. Claims must use the same model, dtype, hardware, prompt/output lengths, sampling parameters, tensor parallel size, and batch constraints. The current matrix has one incomplete workload because Hugging Face preflight OOMed.
- [ ] Add online serving benchmarks with TTFT, TPOT, P50/P95/P99 latency, throughput, memory, queue length, prefix-cache hit rate, and preemption count.
- [ ] Expand correctness tests across all context lengths, block sizes, GQA ratios, head dimensions, batch sizes, prefix cache reuse, chunked prefill, preemption, CUDA Graph, and tensor parallel execution. Core attention and scheduler coverage exists, but the matrix is not complete.
- [ ] Add Transformers alignment beyond sampled prefill logits, including greedy or controlled-token generation once deterministic sampling support exists.

## P0: Serving semantics

- [x] Add request cancellation on client disconnect and explicit cancellation APIs.
- [x] Add timeout handling, maximum queue size, and backpressure for request queues and per-stream output queues.
- [x] Define and expose a clear request state machine: queued, prefill, decode, finished, cancelled, failed.
- [ ] Improve OpenAI compatibility: unsupported parameters should return clear errors, while implemented parameters should include accurate usage and finish reasons.
- [x] Add integration tests for completions, chat completions, streaming, malformed requests, and engine errors.

## P1: Scheduling and KV cache

- [ ] Add configurable scheduling policies such as FCFS, priority, deadline-aware, or throughput-oriented modes.
- [ ] Improve preemption policy to avoid repeated recompute under memory pressure.
- [ ] Add prefix-cache capacity control and eviction, such as LRU for inactive hashed blocks.
- [ ] Add admission control based on remaining KV blocks and token budget.
- [ ] Evaluate CPU/GPU KV swapping as a future alternative to recompute-on-preemption.

## P1: Kernels and execution

- [x] Profile CPU scheduling, kernel launch overhead, attention, GEMM, NCCL, and memory movement with available profiler tooling. Current evidence covers Nano-vLLM eager and CUDA Graph long-context runs; broader cross-engine profiling remains useful.
- [x] Evaluate split-KV or partitioned-softmax decode for very long contexts where one program loops over many blocks. The implementation and benchmark evidence exist; further tuning is still open.
- [x] Add a principled attention backend selector for the implemented varlen, paged, split-KV, and paged-prefill Q-tile paths.
- [x] Implement and evaluate W8A16 weight-only quantization. It is currently eager-only and is an explicit memory-optimization path, not a guaranteed faster default.
- [ ] Measure tensor parallel throughput, scaling efficiency, NCCL overhead, and failure behavior.

## P1: Observability and operations

- [ ] Add `/metrics`, `/health`, and `/ready` with distinct liveness/readiness semantics. `/health` and internal read-only metrics snapshots exist; Prometheus metrics and a separate readiness endpoint do not.
- [ ] Export request count, queue length, batch size, TTFT, TPOT, prefill/decode throughput, KV block usage, prefix-cache hit rate, preemption count, OOM count, and exception count.
- [ ] Add structured logs with request IDs.
- [ ] Provide a Dockerfile or locked environment instructions for reproducible setup.
- [ ] Add CI for linting, CPU tests, optional GPU tests, and API integration tests.

## P2: Model and API expansion

- [ ] Introduce a model registry and weight-loader abstraction instead of directly instantiating Qwen3.
- [ ] Add at least one additional architecture, such as Llama-like models, with documented compatibility tests.
- [ ] Expand sampling with greedy mode, top-k, top-p, repetition penalty, stop strings, seed control, and logprobs where feasible.
- [ ] Evaluate LoRA loading, quantized model loading, and multi-model routing after the core evidence base is stronger.

## Documentation rules

- README should only advertise capabilities that are implemented or explicitly marked as roadmap.
- Performance claims must link to [docs/baseline.md](docs/baseline.md), [docs/benchmark.md](docs/benchmark.md), or raw benchmark artifacts.
- Do not describe Nano-vLLM as matching or exceeding vLLM unless a same-condition benchmark exists.
- Keep architecture and roadmap updates in sync with code changes and [docs/AI_INFRA_COMPLETED.md](docs/AI_INFRA_COMPLETED.md).
