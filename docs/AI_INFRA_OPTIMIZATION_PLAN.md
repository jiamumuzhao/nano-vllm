# Nano-vLLM AI Infra Optimization Plan

Goal: evolve Nano-vLLM from a readable inference-engine prototype into a credible AI-infra portfolio project with accurate architecture documentation, reproducible evidence, and honest capability boundaries.

## 0. Current baseline as of 2026-07-20

- Model: Qwen3-0.6B.
- Hardware used for historical baseline: 2 x NVIDIA RTX 2080 Ti.
- Implemented capabilities: paged KV cache, prefix caching, chunked prefill, preemption, Triton varlen/paged attention, CUDA Graph, Tensor Parallel, AsyncEngine, OpenAI-style API, and SSE streaming.
- Current correctness baseline: `/opt/anaconda3/bin/python -m pytest -q`, `20 passed`.
- Historical performance record: under 8 concurrent requests with 128 input tokens and 64 output tokens, Nano-vLLM CUDA Graph reached about `1410 tok/s` end-to-end output throughput, while Nano-vLLM eager reached about `306 tok/s`.
- Historical performance record is not a same-condition comparison against vLLM or Hugging Face generate. A new same-condition offline benchmark is recorded in [docs/benchmarks/offline_comparison_strict_2026-07-12.md](benchmarks/offline_comparison_strict_2026-07-12.md), and a 3x3 offline workload matrix is recorded in [docs/benchmarks/offline_matrix_2026-07-12.md](benchmarks/offline_matrix_2026-07-12.md).

## 1. P0: Build trustworthy correctness and performance evidence

### 1.1 Unified benchmark protocol

- [x] Fix model, GPU, PyTorch/CUDA/Triton versions, dtype, max model length, sampling parameters, warmup, and repetition count for the strict default offline workload with 10 measured runs.
- [x] Complete one 3x3 offline matrix across concurrency `1/8/16` and input length `128/512/2048` with output length `64`; eight formal four-engine workloads completed and the `2048x16` workload is recorded as incomplete due to Hugging Face OOM.
- [ ] Cover larger production-oriented matrices such as concurrency `1/8/32/128`, input length `128/1K/4K/8K`, and output length `32/128/512`.
- [x] Provide and validate a reproducible online AsyncEngine smoke command/schema for TTFT, TPOT, end-to-end throughput, P50/P95/P99 latency, peak memory, KV usage, prefix-cache hit/token-hit rate, preemption count, and request outcomes. The verified RTX 2080 Ti/Qwen3-0.6B smoke uses `gpu_memory_utilization=0.9`; in this environment `0.25` reaches runtime KV allocation failure and is not treated as evidence. These are current-environment observations, not cross-GPU minimums. The current AsyncEngine cannot expose an unambiguous prefill/decode wall-time boundary, so those two tok/s fields are explicitly null with a reason rather than guessed.
- [ ] Record the complete production-oriented matrix with real GPU results, including concurrency `1/8/32/128`, input length `128/1K/4K/8K`, output length `32/128/512`, and all requested phase metrics.
- [x] Emit machine-readable JSONL and Markdown summary tables for the strict default offline workload and the 3x3 offline matrix.

Acceptance: any result can be reproduced by one command and includes environment plus at least three measured repetitions.

### 1.2 Same-condition baselines

- [x] Compare Hugging Face generate, Nano-vLLM eager, Nano-vLLM CUDA Graph, and vLLM under the same strict default offline model, hardware, prompt/output lengths, sampling config, max sequence/batched-token constraints, and measured seeds.
- [x] Compare the same four engines in a bounded 3x3 offline matrix and document the incomplete `2048x16` workload without treating it as a complete comparison.
- [x] Separate offline throughput from online serving benchmarks in docs and benchmark reports.
- [x] Remove unsupported README claims that imply vLLM parity.

Acceptance: every performance conclusion links to commands, raw data, and environment metadata.

### 1.3 Correctness and regression

- [x] Add attention tests across context length, block size, GQA ratio, head dimension, and batch size. Coverage includes paged prefill/decode, Split-KV contexts through 16K, mixed-length batches, 16/256-token KV blocks, MHA/GQA, and head dimensions 64/128/256.
- [ ] Add Transformers per-token logits or deterministic generation alignment. The TP=1 Qwen3-0.6B GQA verifier now performs a full-model v3 execution-order bisect, but the latest formal GPU artifact `docs/benchmarks/logits_alignment_qwen3_tp1.jsonl` still fails the strict gate (max abs `0.1152344`, mean `0.0094781`, minimum top-1 `0.875`). The first case-0 prefill divergence is `layer_1.k_norm_output`, immediately after `layer_1.q_norm_output` (max `0.125`, mean `0.00068061`, RMSE `0.00416488`, top-1 `1.0`; KV head `2`, head-dim `52`). The v3 record shows equal `[1,1,8,128]` shapes, `dim=-1` normalization, and an upstream K-norm input max error of `0.00048828125`; exact FP16 alignment remains a blocker, not evidence of cross-head RMSNorm. Production FP16 keeps the original merged gate/up path; split-gates is validation-only. Artifacts: `docs/benchmarks/logits_alignment_qwen3_tp1_layerwise_v3.jsonl` plus preserved v1/v2 evidence.
- [x] Add GPU Engine E2E regressions for prefix cache, chunked prefill, preemption/recompute, CUDA Graph/eager equivalence, and TP=2. These run only with CUDA and the local Qwen3-0.6B model; missing CUDA/model paths are explicit skips, not CPU coverage. They inspect token ids, scheduler events and KV block state rather than generated text.
- [x] Define performance regression thresholds for benchmark runs. The current implementation is an offline JSONL benchmark gate with initial protection thresholds; online serving indicators remain tracked under 1.1.

Acceptance: CPU and GPU tests are separated, and critical-path changes can catch numerical or performance regressions. The P0 lifecycle items are considered complete only when the corresponding GPU E2E tests actually pass; a missing CUDA/model path is an explicit skip, not CPU coverage.

## 2. P0: Serving production semantics

- [x] Support client disconnect cancellation, explicit cancellation, timeout, and max queue size.
- [x] Add backpressure for the intake queue, scheduler queues, and per-request streaming queues.
- [x] Define request states: queued, prefill, decode, finished, cancelled, failed.
- [x] Add request status/cancellation endpoints and CPU integration tests for completions, chat completions, streaming, queue-full errors, and cleanup.
- [x] Add UUID request IDs and precise finish reasons (`stop`, `cancelled`, `timeout`, `error`, `backpressure`).

Acceptance evidence: `tests/test_async_engine.py` and `tests/test_api_serving.py` cover
cancellation, timeout, bounded intake/stream queues, request-level failure isolation,
engine-level fatal handling, leak-free terminal streams, shutdown cleanup, request status,
completion/chat-completion success and SSE, malformed request validation, HTTP 503/429
behavior, and KV-release hooks without CUDA or a model. GPU scheduling remains
covered by the existing scheduler/block-manager and E2E suites. Defaults are
`max_queue_size=256`, `request_timeout_s=300`, and `stream_queue_size=16`.

## 3. P1: Scheduling and KV cache depth

- [ ] Add configurable scheduling policies and evaluate fairness/throughput tradeoffs.
- [ ] Measure and optimize preemption to avoid repeated recomputation under pressure.
- [ ] Add prefix-cache capacity control and eviction policy.
- [ ] Add admission control based on remaining KV blocks and token budget.
- [ ] Evaluate KV swapping only after recompute-preemption behavior is measured.

## 4. P1: Kernels and execution optimization

- [x] Profile Nano-vLLM long-context prefill/decode behavior with CPU/CUDA profiler evidence for short vs long context in eager and CUDA Graph modes. Evidence: [docs/profiling_long_context.md](profiling_long_context.md).
- [x] Evaluate split-KV / partitioned-softmax paged decode for long contexts. The eager implementation supports 1/2/4/8/16/32 partition buckets with preallocated workspace, correctness coverage through 16K context, and a direct decode benchmark at `scripts/benchmark_decode_split_kv.py`.
- [x] Add a measured Q-tile paged-prefill path selector. `serial`, `parallel`, and `auto` modes preserve legacy boolean compatibility; the current RTX 2080 Ti/Qwen3 eager matrix found no clear end-to-end parallel gain, so `auto` safely selects serial. Evidence: `docs/benchmarks/qwen3_eager_q_tile_20260720T032305Z.md`.
- [ ] Implement W8A16 first, then evaluate FP8 KV cache with accuracy, memory, throughput, and GPU support evidence.
- [ ] Complete TP benchmark evidence: throughput, scaling efficiency, NCCL profile, and failure handling.

## 5. P1: Engineering and maintainability

- [ ] Add CI for lint, CPU tests, optional GPU tests, and API integration tests.
- [ ] Provide Dockerfile or locked setup instructions.
- [ ] Improve configuration validation and user-facing error messages.
- [x] Align README, architecture documentation, benchmark evidence, and roadmap with current code state.

Acceptance: a new developer can install, test, serve, and benchmark the project from documented commands.

## 6. P2: Boundaries and extensions

- [ ] Publish an explicit compatibility matrix for the current Qwen3 path.
- [ ] Add at least one additional model architecture with loading, generation, serving, and correctness tests.
- [ ] Evaluate LoRA, quantized model loading, and multi-model routing after core evidence is complete.

## 7. Documentation and portfolio deliverables

- [x] README includes project positioning, architecture diagram, feature list, quick start, API example, validation/performance caveats, and documentation navigation.
- [x] [docs/architecture.md](architecture.md) explains request lifecycle, scheduler, KV cache, attention, execution, serving, and limits.
- [x] [docs/benchmark.md](benchmark.md) records historical evidence, the strict single-workload comparison, the 3x3 offline matrix, long-context profiling evidence, unmeasured gaps, and a future benchmark protocol.
- [x] [OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md) separates completed capabilities from future work.
- [ ] Write short reports for major future optimizations only after they have before/after data.
- [ ] Keep resume claims limited to metrics with commands and raw data.

Suggested resume wording:

> Implemented a lightweight LLM inference engine from scratch with paged KV cache, prefix caching, chunked prefill, preemptive continuous batching, Triton GQA-aware varlen/paged attention, CUDA Graph decode replay, NCCL tensor parallelism, and OpenAI-style SSE serving. Maintained reproducible correctness and benchmark documentation with explicit limits on unsupported comparisons.
