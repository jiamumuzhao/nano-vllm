# Nano-vLLM Profile - eager input128 seqs8

- Created at: `2026-07-12T08:54:46+00:00`
- Command: `/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode eager --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20`
- Prompt SHA-256: `a412c076e2a785e01d46fb9b53c33868832776b9930fc50d7b5ec325d8cc9de1`

## Metrics

| Metric | Value |
| --- | ---: |
| Initialization seconds | 77.7197 |
| Prefill tokens | 1024 |
| Prefill synchronized seconds | 0.0986 |
| Prefill tok/s | 10380.35 |
| Decode tokens | 504 |
| Decode synchronized seconds | 2.2142 |
| Decode tok/s | 227.62 |
| Decode step mean seconds | 0.0351 |
| Decode step median seconds | 0.0351 |
| Decode step p95 seconds | 0.0355 |
| Max CUDA allocated bytes | 3805318656 |
| Max CUDA reserved bytes | 3869245440 |

## Top CUDA Events

| Event | Count | CUDA total ms | CPU total ms |
| --- | ---: | ---: | ---: |
| `aten::linear` | 7232 | 188.227 | 277.203 |
| `aten::matmul` | 7232 | 188.227 | 227.617 |
| `aten::mm` | 7232 | 188.227 | 214.399 |
| `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8::Params)` | 7056 | 131.243 | 0.000 |
| `_varlen_prefill_kernel` | 28 | 72.389 | 0.000 |
| `_decode_paged_kernel` | 1764 | 36.977 | 0.000 |
| `turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn` | 64 | 36.546 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 17.343 | 165.564 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 17.343 | 0.000 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 56 | 11.408 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 10.015 | 17.494 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 10.015 | 14.772 |
| `Torch-Compiled Region: 2/1` | 3584 | 8.957 | 360.041 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 3584 | 8.957 | 217.492 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 8.404 | 1.520 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 8.404 | 0.000 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 3584 | 8.278 | 185.328 |
| `Torch-Compiled Region: 0/3` | 3584 | 8.276 | 332.333 |
| `Torch-Compiled Region: 1/1` | 1792 | 7.297 | 235.186 |
| `## Call CompiledFxGraph fbhs6ymcy6jox7vu4d6dkickoul4rv6ilylifbsxochsg5stucq3 ##` | 1792 | 7.297 | 162.550 |

## Top CPU Events

| Event | Count | CPU total ms | CUDA total ms |
| --- | ---: | ---: | ---: |
| `Torch-Compiled Region: 2/1` | 3584 | 360.041 | 8.957 |
| `Torch-Compiled Region: 0/3` | 3584 | 332.333 | 8.276 |
| `aten::linear` | 7232 | 277.203 | 188.227 |
| `Torch-Compiled Region: 1/1` | 1792 | 235.186 | 7.297 |
| `aten::matmul` | 7232 | 227.617 | 188.227 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 3584 | 217.492 | 8.957 |
| `aten::mm` | 7232 | 214.399 | 188.227 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 3584 | 185.328 | 8.278 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 165.564 | 17.343 |
| `## Call CompiledFxGraph fbhs6ymcy6jox7vu4d6dkickoul4rv6ilylifbsxochsg5stucq3 ##` | 1792 | 162.550 | 7.297 |
| `Torch-Compiled Region: 3/1` | 1792 | 158.482 | 4.217 |
| `cuLaunchKernel` | 19984 | 131.280 | 0.023 |
| `TorchDynamo Cache Lookup` | 10880 | 116.807 | 0.000 |
| `Pregraph bytecode` | 10880 | 102.645 | 0.000 |
| `## Call CompiledFxGraph fpebhi5jbqvor5ofgvfrf45rcn2secgiisliut6sb7bvdtdijqwu ##` | 1792 | 89.053 | 4.217 |
| `aten::is_nonzero` | 28 | 58.321 | 0.026 |
| `aten::item` | 28 | 58.292 | 0.026 |
| `aten::_local_scalar_dense` | 28 | 58.255 | 0.026 |
| `cudaStreamSynchronize` | 92 | 57.996 | 0.000 |
| `AOTDispatcher Runtime Wrapper Prologue` | 10880 | 43.019 | 0.000 |

