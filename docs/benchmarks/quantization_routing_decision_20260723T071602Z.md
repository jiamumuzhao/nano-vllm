# Quantization production routing decision

This is an offline decision record; runtime routing does not read benchmark JSON.

- Source benchmark: `docs/benchmarks/w8a8_break_even_20260723T030044Z.json`
- W8A8 eligible shapes: **0**
- GPU: NVIDIA GeForce RTX 2080 Ti

## Production routing

| Decision | Value |
|---|---|
| Default performance route | `fp16` (`quantization=none`) |
| Explicit memory route | `w8a16` |
| W8A8 production enabled | `false` |

W8A16 is an explicit weight-memory optimization mode. The RTX 2080 Ti benchmark does not guarantee decode speedup over FP16.
W8A8 WMMA and cuBLASLt remain benchmark-only; no W8A8 model end-to-end result is claimed.

## Reconsideration conditions

- A reproducible benchmark must pass the existing W8A16/FP16 integration gate.
- The passing result must be repeated on the target hardware and workload matrix.
- A real model end-to-end validation must be completed after the operator result.
- Only then may production routing be reconsidered; this record does not change runtime routing.
