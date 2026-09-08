# Serving Benchmark Analysis

Input: docs/benchmarks/latest/online_policy_matrix_latest.jsonl

Rows are grouped by scheduling policy and preemption parameters. Latency and throughput are arithmetic means across measured workload records.

| rank | policy | cooldown | max preempts | runs | success | output tok/s | TTFT p95 (s) | E2E p95 (s) | preemptions | recompute tokens | deferred |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | throughput | 2 | 1 | 12 | 100.00% | 118.3112 | 2.3472 | 4.8403 | 0.0000 | 0.0000 | 0.0000 |
| 2 | latency | 2 | 1 | 12 | 100.00% | 116.6598 | 2.3789 | 4.9264 | 0.0000 | 0.0000 | 0.0000 |
| 3 | fcfs | 2 | 1 | 12 | 100.00% | 113.7240 | 2.5103 | 5.1720 | 0.0000 | 0.0000 | 0.0000 |

Recommended first candidate: throughput with cooldown 2 and max preempts 1. Validate this candidate with repeated runs before making it the deployment default.
