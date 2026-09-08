# Optimization Memory

> Purpose: accumulate optimization decisions, evidence, and follow-up work.
> Small changes should be recorded here first and committed in batches instead
> of creating a local commit for every incremental adjustment.

## Current status

Last updated: 2026-08-21

### Completed

- Online serving benchmark:
  - Added reproducible AsyncEngine benchmark coverage.
  - Completed the concurrency/input-length matrix.
  - Identified the 2048-token failure as a max_model_len capacity/configuration
    issue rather than an immediate GPU OOM.
  - Re-ran the 2048-token workloads with max_model_len=4096.

- Workload parameter validation:
  - Validate that max_model_len covers input plus output tokens.
  - Validate alignment with the KV block size.
  - Fail early with an actionable error message.

- Prefix Cache capacity control:
  - Added configurable prefix_cache_max_blocks.
  - -1 keeps all available KV blocks eligible for prefix caching.
  - Inactive cached blocks use LRU eviction.
  - Active blocks (ref_count > 0) are protected from eviction.
  - Added cache size, capacity ratio, and eviction counters to scheduler metrics.
  - Exposed the limit through the OpenAI server and serving benchmark CLI.

- Preemption state semantics:
  - Preempted requests release their block references and clear their block table.
  - They return to the waiting queue with explicit SequenceStatus.QUEUED.
  - They resume from Prefill when scheduled again.

- Sequence state simplification:
  - Kept six canonical states only:
    QUEUED, PREFILL, DECODE, FINISHED, CANCELLED, FAILED.
  - Removed the redundant WAITING and RUNNING aliases.

## Design constraints

- Releasing a request's blocks does not require clearing physical GPU KV memory.
  The contents may be reused by Prefix Cache or overwritten when new KV values
  are computed.
- A shared block is released only after its reference count reaches zero.
- Prefix Cache eviction must never invalidate an active sequence.
- Do not push remote changes automatically.
- Do not create a local commit for every small edit; batch related changes.

## Verification

- Python syntax compilation passed for the changed modules.
- Prefix Cache capacity tests were added for:
  - LRU eviction of inactive blocks.
  - Protection of active blocks.
- Preemption state test was added.
- Full pytest execution on the server experienced unusually long periods with
  no output and was interrupted; this remains a verification follow-up.

## Next candidates

1. Improve preemption fairness and prevent starvation of new requests.
2. Add a scheduler-level test for repeated preemption under KV pressure.
3. Run the full test suite when the server test environment is responsive.
4. Benchmark Prefix Cache hit rate, eviction rate, latency, and throughput under
   several capacity limits.


## New work in progress

- Request-level preemption policy:
  - Selects a victim using prior preemption count, running age, and KV Block footprint.
  - Protects requests that have already been preempted from repeated immediate eviction.
  - Uses waiting age instead of append-left ordering when admitting queued requests.
  - Records victim KV footprint and per-request preemption count in events.
  - Full test-suite verification remains pending because the remote pytest
    environment previously stalled without output.


- KV Block allocation order:
  - Added a plain-free queue and kept the existing all-free queue for
    compatibility and capacity accounting.
  - Plain free Blocks are allocated before inactive Prefix Cache Blocks.
  - When plain free Blocks are exhausted, the Prefix Cache LRU head is
    explicitly evicted and reused.
  - Added free KV Block breakdown metrics.


- Added a BlockManager invariant checker and regression coverage for queue
  partitioning, cache eviction, and reuse transitions.


- Production observability:
  - Added a dependency-free Prometheus text endpoint at GET /metrics.
  - Exposes request lifecycle gauges, accepted/finished/cancelled/failed
    counters, prompt/generation token counters, TTFT and E2E latency sums/counts.
  - Exposes KV free/used/peak capacity, plain-vs-cached free blocks, Prefix
    Cache hit/eviction metrics, and preemption count.
  - Added request timestamps for first-token and terminal latency accounting.


- Scheduler/KV hot-path optimization:
  - Replaced typed free-block deques with indexed ordered sets supporting O(1)
    removal, append, and pop-left.
  - Preserved the compatibility free-block view and added synchronization for
    tests or tools that replace it directly.


## 2026-08-21: waiting queue heap optimization (uncommitted)

- Replaced Scheduler.waiting deque plus min scan with a stable min-heap. Enqueue and dequeue are O(log n), avoiding an O(n) aging scan on every scheduler step.
- The key remains (queued_step, arrival_order), preserving aging fairness for new and preempted requests. Partially prefilling sequences refresh their current-round key so they cannot consume another slot in the same round.
- Cancellation and failure use lazy remove instead of rebuilding the full waiting queue. clear() keeps engine shutdown compatible.
- Verification: tests/test_scheduler.py 5 passed; tests/test_block_manager.py 7 passed.
- Not committed or pushed. Next step: benchmark scheduler step latency under high concurrency and monitor lazy heap entry accumulation.


## 2026-08-21: waiting heap compaction (uncommitted)

- Added threshold-based heap rebuild after lazy removals: rebuild only when heap size exceeds twice the active entry count plus 64.
- Cancellation remains O(1) in the common path; compaction is amortized and prevents stale entries from growing without bound under cancellation/preemption churn.
- Verification: tests/test_scheduler.py 6 passed; Python syntax and git diff check passed.


## 2026-08-21: scheduler queue benchmark (uncommitted)

- Added scripts/benchmark_scheduler_queue.py, a CPU-only benchmark comparing the waiting heap against the previous deque + min/remove scan.
- Sample result on Ubuntu host, 1000 operations x 5 samples:
  - size 32: heap 0.49 us/op vs scan 1.60 us/op
  - size 128: heap 0.53 us/op vs scan 5.88 us/op
  - size 512: heap 0.61 us/op vs scan 24.49 us/op
  - size 2048: heap 0.72 us/op vs scan 94.55 us/op
- At queue size 2048, the heap achieved about 131x higher queue-operation throughput. This is a CPU queue microbenchmark, not end-to-end model serving throughput.


## 2026-08-21: configurable scheduling policies (uncommitted)

- Added scheduling_policy with fcfs as the backward-compatible default.
- fcfs preserves aging order; latency uses shortest estimated prompt-plus-generation work first; throughput prefers smaller estimated KV footprint so more requests can coexist in a batch.
- Exposed the policy through Config, the OpenAI server CLI, and scripts/benchmark_serving.py; scheduler metrics now report the active policy.
- Added policy validation and scheduler tests. Related scheduler, metrics, and API tests: 35 passed.
- Policy trade-off: latency/throughput modes are workload heuristics and should be evaluated with the online serving matrix before becoming deployment defaults.


## 2026-08-21: KV admission control (uncommitted)

- Added conservative request admission based on prompt tokens, max generation tokens, block size, current free KV blocks, and max model length.
- Requests that can never fit are rejected with a clear 400-level admission error; requests that temporarily exceed available KV capacity remain in the intake queue and are retried after active work releases capacity.
- Added per-request-batch KV reservation during intake draining so multiple requests cannot all pass against the same pre-allocation free-block snapshot.
- Added admitted/deferred/rejected counters to service metrics and Prometheus output.
- Verification: AsyncEngine, scheduler, metrics, and API tests pass; the focused AsyncEngine/API run passed 19 tests.
- This is intentionally conservative and does not yet account for prefix-cache reuse; online serving benchmarks should validate whether the policy over-defers long requests.


## 2026-08-21: prefix-aware admission (uncommitted)

- Added BlockManager.count_cached_prefix_blocks() for read-only complete-prefix lookup.
- Admission now subtracts reusable Prefix Cache blocks from the estimated KV requirement, reducing over-deferral for repeated long prompts.
- Verification: focused BlockManager, AsyncEngine, Scheduler, Metrics, and API tests passed 51 tests.
- Remaining trade-off: the estimate still reserves for the requested maximum generation length, so it remains safe but conservative.


## 2026-08-21: online policy matrix benchmark (uncommitted)

- Added --scheduling-policy-matrix to scripts/benchmark_serving.py.
- One invocation now runs fcfs, throughput, and latency over the same concurrency/input/output workloads, with policy labels in JSONL and Markdown.
- Benchmark rows now include admission A/D/R counters alongside TTFT, TPOT, E2E latency, output throughput, KV peak, and Prefix Cache hit rates.
- Example: python scripts/benchmark_serving.py --model MODEL --scheduling-policy-matrix --concurrencies 1,4,8,16 --input-lens 128,512,2048 --output-lens 64.
- Verification: serving metrics and API tests passed 26 tests; benchmark module compiles and git diff check passes.


## 2026-08-21: preemption hysteresis (uncommitted)

- Added configurable preemption_cooldown_steps (default 2) and max_preemptions_per_step (default 1).
- Victim selection filters recently preempted sequences when alternatives exist, reducing repeated recompute of the same request.
- Added recompute token accounting and cooldown-skip counters to scheduler snapshots and Prometheus metrics.
- Verification: scheduler, metrics, and API tests passed 37 tests.
- The defaults are intentionally conservative; the online policy matrix should measure whether cooldown improves P99 without harming GPU utilization.

## 2026-08-21: tunable preemption controls (uncommitted)

- Exposed preemption_cooldown_steps and max_preemptions_per_step through the OpenAI server CLI and online serving benchmark.
- Every benchmark JSONL record now carries the effective preemption parameters, making cooldown comparisons reproducible.
- Example: --preemption-cooldown-steps 2 --max-preemptions-per-step 1.
- Verification: scheduler, metrics, and API tests passed 37 tests.


## 2026-08-21: readiness probe (uncommitted)

- Added AsyncEngine.is_ready() and a separate GET /ready endpoint.
- /health reports process liveness; /ready returns 503 until the engine loop is started or after engine failure/shutdown.
- This closes a production-serving gap relative to the baseline offline-only API.
- Verification: API, AsyncEngine, and scheduler tests passed 30 tests.


## 2026-08-21: preemption parameter sweep benchmark (uncommitted)

- Added benchmark parameters preemption-cooldown-grid and max-preemptions-per-step-grid.
- The serving matrix now iterates policy x cooldown x per-step limit x workload, with each run carrying the effective values in JSONL.
- Example: --scheduling-policy-matrix --preemption-cooldown-grid 0,2,4 --max-preemptions-per-step-grid 1,2.
- Verification: benchmark compiles, CLI grid parsing works, and git diff check passes.


## 2026-08-21: serving benchmark analyzer (uncommitted)

- Added scripts/analyze_serving_benchmark.py.
- Groups measured JSONL records by scheduling policy, cooldown, and max preemptions per step.
- Reports success rate, output throughput, TTFT P95, E2E P95, preemption count, recompute tokens, and deferred requests.
- Produces a ranked Markdown summary with a first candidate recommendation while clearly requiring repeated validation.
- Verified with a synthetic record and Python compilation.


## 2026-08-21: serving benchmark quality gate (uncommitted)

- Added threshold checks to scripts/analyze_serving_benchmark.py for minimum success rate, minimum throughput, and maximum TTFT/E2E P95.
- The analyzer now returns exit code 1 and prints failed configurations when a threshold is violated, so it can be used in CI or release validation.
- Verified against an existing 66.67% success-rate record with a 90% threshold; the gate returned exit code 1.


## 2026-08-21: latest GPU online benchmark

- Ran the latest uncommitted code on 2x NVIDIA GeForce RTX 2080 Ti with Qwen3-0.6B, TP=2, max_model_len=4096, max_num_seqs=16, and max_num_batched_tokens=4096.
- Workload matrix: concurrency 1/4/8/16, input 128/512/2048, output 64; FCFS, throughput, and latency policies; 36 measured records total.
- All 36 records completed successfully with cleanup diagnostics; no OOM or runtime allocation failure.
- Aggregated result: throughput policy 118.31 output tok/s, latency policy 116.66, FCFS 113.72. Mean TTFT P95: 2.347s, 2.379s, 2.510s respectively; mean E2E P95: 4.840s, 4.926s, 5.172s.
- Quality gate passed with success rate 100%, throughput >=100 tok/s, TTFT P95 <=3s, and E2E P95 <=6s.
- Artifacts: docs/benchmarks/latest/online_policy_matrix_latest.jsonl, online_policy_matrix_latest.md, and online_policy_matrix_analysis.md.


## 2026-08-21: structured request event logs (uncommitted)

- Added low-frequency JSON lifecycle events for request_admitted, request_deferred, request_rejected, and request_terminal.
- Events include request ID, sequence ID, prompt/generation counts, reason, required blocks, and E2E duration where applicable. Token decode hot path is not logged.
- Added --log-level to the OpenAI server CLI; default WARNING keeps production noise low, while INFO enables event streams.
- Verification: AsyncEngine, API, scheduler, and metrics tests passed 46 tests.


## 2026-08-21: TP scaling benchmark tool (uncommitted)

- Added scripts/benchmark_tp_scaling.py to run identical online serving workloads at TP=1/2 and calculate output throughput, TTFT/E2E P95, success rate, and scaling efficiency.
- The tool writes per-TP JSONL/Markdown plus summary.json and summary.md.
- Verified with synthetic records; actual TP scaling run remains pending.
