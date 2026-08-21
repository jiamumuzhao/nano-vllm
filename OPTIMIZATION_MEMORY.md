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
