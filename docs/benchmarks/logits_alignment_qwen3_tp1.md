# Nano-vLLM vs Transformers logits alignment

This is a TP=1 correctness verification, not a performance benchmark. The Qwen3-0.6B model uses GQA; MHA is covered by synthetic attention tests.

- Model: `/root/huggingface/Qwen3-0.6B`
- GPU: `NVIDIA GeForce RTX 2080 Ti`; torch `2.11.0+cu130`; CUDA `13.0`
- Thresholds: max abs `<= 0.05`, mean abs `<= 0.005`, top-1 `== 1.0`
- Overall: **FAIL**

| Case | Phase | Rows | Elements | Max abs | Mean abs | RMSE | Top-1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[1]` | prefill | 1 | 151936 | 0.03125 | 0.00527601 | 0.00645739 | 1.000000 | FAIL |
| `[1]` | decode | 8 | 1215488 | 0.115234 | 0.00947815 | 0.0136995 | 1.000000 | FAIL |
| `[16]` | prefill | 16 | 2430976 | 0.078125 | 0.00514142 | 0.00715709 | 1.000000 | FAIL |
| `[16]` | decode | 8 | 1215488 | 0.0334473 | 0.0048169 | 0.00602723 | 1.000000 | PASS |
| `[17]` | prefill | 17 | 2582912 | 0.0749512 | 0.00421594 | 0.00558827 | 1.000000 | FAIL |
| `[17]` | decode | 8 | 1215488 | 0.03125 | 0.00526431 | 0.00655689 | 1.000000 | FAIL |
| `[31]` | prefill | 31 | 4710016 | 0.0749512 | 0.00411727 | 0.00540005 | 1.000000 | FAIL |
| `[31]` | decode | 8 | 1215488 | 0.0820312 | 0.00535828 | 0.00770227 | 1.000000 | FAIL |
| `[32]` | prefill | 32 | 4861952 | 0.0749512 | 0.00412162 | 0.0053883 | 1.000000 | FAIL |
| `[32]` | decode | 8 | 1215488 | 0.0361328 | 0.00548503 | 0.007107 | 1.000000 | FAIL |
| `[128]` | prefill | 128 | 19447808 | 0.0673828 | 0.00398938 | 0.00520639 | 1.000000 | FAIL |
| `[128]` | decode | 8 | 1215488 | 0.0385742 | 0.00452289 | 0.005767 | 0.875000 | FAIL |
| `[1024]` | prefill | 1024 | 155582464 | 0.0858536 | 0.00372389 | 0.00480229 | 0.997070 | FAIL |
| `[1024]` | decode | 8 | 1215488 | 0.0341797 | 0.00428172 | 0.00541351 | 1.000000 | PASS |
| `[17, 32, 127]` | prefill | 176 | 26740736 | 0.0561523 | 0.00412659 | 0.00546903 | 1.000000 | FAIL |
| `[17, 32, 127]` | decode | 24 | 3646464 | 0.0449219 | 0.00482885 | 0.00626198 | 1.000000 | PASS |

The decode rows use the same deterministic continuation token ids on both implementations; mixed-batch padding is excluded by construction.
First full-model boundary over max-error threshold: `final_norm.` (case=0, phase=decode, max_abs=0.1875, mean_abs=0.00993023, top1=1.000000).
Layerwise v2 applies strict equal-shape packed/padded validation; its contract and records are written to `logits_alignment_qwen3_tp1_layerwise_v2.jsonl`.
