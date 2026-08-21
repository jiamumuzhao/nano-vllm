# Nano-vLLM vs Transformers logits alignment

This is a TP=1 correctness verification, not a performance benchmark. The Qwen3-0.6B model uses GQA; MHA is covered by synthetic attention tests.

- Model: `/root/huggingface/Qwen3-0.6B`
- GPU: `NVIDIA GeForce RTX 2080 Ti`; torch `2.11.0+cu130`; CUDA `13.0`
- Thresholds: max abs `<= 0.05`, mean abs `<= 0.005`, top-1 `== 1.0`
- Overall: **FAIL**

| Case | Phase | Rows | Elements | Max abs | Mean abs | RMSE | Top-1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[1]` | prefill | 1 | 151936 | 0.03125 | 0.00527601 | 0.00645739 | 1.000000 | FAIL |
| `[1]` | decode | 4 | 607744 | 0.03125 | 0.00458814 | 0.00589068 | 1.000000 | PASS |
| `[1]` | prefill | 1 | 151936 | 0.03125 | 0.00527601 | 0.00645739 | 1.000000 | FAIL |
| `[1]` | decode | 4 | 607744 | 0.0273438 | 0.0043741 | 0.00554878 | 1.000000 | PASS |

The decode rows use the same deterministic continuation token ids on both implementations; mixed-batch padding is excluded by construction.
First full-model boundary over max-error threshold: `final_norm.` (case=0, phase=decode, max_abs=0.124023, mean_abs=0.00427029, top1=1.000000).
Layerwise v2 applies strict equal-shape packed/padded validation; its contract and records are written to `logits_alignment_qwen3_tp1_layerwise_v2.jsonl`.
