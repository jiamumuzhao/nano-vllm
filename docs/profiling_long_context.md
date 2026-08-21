# Long-Context Profiling

The long-context profiling phase records Nano-vLLM CPU/CUDA profiler evidence for the short-context and long-context batch-8 workloads used in the strict offline matrix.

- Detailed profile report: [profiles/long_context_profile_2026-07-12.md](profiles/long_context_profile_2026-07-12.md)
- Machine-readable profile summary: [profiles/long_context_profile_2026-07-12.json](profiles/long_context_profile_2026-07-12.json)
- Matrix benchmark reference: [benchmarks/offline_matrix_2026-07-12.md](benchmarks/offline_matrix_2026-07-12.md)

Main finding: Nano-vLLM's long-context graph workload is dominated by paged-prefix prefill and by a paged decode kernel whose per-step latency grows with context length. This supports prioritizing split-KV / partitioned-softmax paged decode and paged-prefill/block-table access profiling next; no optimization is implemented here.

This is offline profiling only. It does not measure TTFT, TPOT, P95/P99 service latency, dynamic arrivals, or vLLM internals.
