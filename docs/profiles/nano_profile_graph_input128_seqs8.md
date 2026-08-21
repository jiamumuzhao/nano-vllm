# Nano-vLLM Profile - graph input128 seqs8

- Created at: `2026-07-12T08:56:31+00:00`
- Command: `/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode graph --num-seqs 8 --input-len 128 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20`
- Prompt SHA-256: `a412c076e2a785e01d46fb9b53c33868832776b9930fc50d7b5ec325d8cc9de1`

## Metrics

| Metric | Value |
| --- | ---: |
| Initialization seconds | 79.2508 |
| Prefill tokens | 1024 |
| Prefill synchronized seconds | 0.0994 |
| Prefill tok/s | 10302.04 |
| Decode tokens | 504 |
| Decode synchronized seconds | 0.2904 |
| Decode tok/s | 1735.30 |
| Decode step mean seconds | 0.0046 |
| Decode step median seconds | 0.0046 |
| Decode step p95 seconds | 0.0047 |
| Max CUDA allocated bytes | 3813858304 |
| Max CUDA reserved bytes | 3894411264 |

## Top CUDA Events

| Event | Count | CUDA total ms | CPU total ms |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8::Params)` | 7056 | 132.319 | 0.000 |
| `_varlen_prefill_kernel` | 28 | 72.946 | 0.000 |
| `aten::linear` | 176 | 54.035 | 6.616 |
| `aten::matmul` | 176 | 54.035 | 5.472 |
| `aten::mm` | 176 | 54.035 | 5.163 |
| `turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn` | 64 | 37.032 | 0.000 |
| `_decode_paged_kernel` | 1764 | 35.725 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 18.583 | 0.000 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 56 | 11.446 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 9.504 | 17.523 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 9.504 | 14.695 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 7.956 | 1.501 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 7.956 | 0.000 |
| `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 56 | 5.557 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_0` | 1792 | 4.863 | 0.000 |
| `triton_poi_fused_mul_silu_split_0` | 1792 | 4.285 | 0.000 |
| `store_kvcache_kernel` | 1792 | 4.116 | 0.000 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, __half, false, __half, __half, __half, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __half const*, __half const*, __half*, __half*, float const*, float const*, __half const*, __half const*, __half*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | 1764 | 3.895 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_1` | 1792 | 3.841 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 113 | 1.500 | 2.694 |

## Top CPU Events

| Event | Count | CPU total ms | CUDA total ms |
| --- | ---: | ---: | ---: |
| `aten::to` | 715 | 230.545 | 0.166 |
| `aten::copy_` | 654 | 230.469 | 0.439 |
| `aten::_to_copy` | 394 | 229.862 | 0.166 |
| `cudaMemcpyAsync` | 681 | 226.776 | 0.000 |
| `aten::is_nonzero` | 28 | 59.075 | 0.026 |
| `aten::item` | 28 | 59.046 | 0.026 |
| `aten::_local_scalar_dense` | 28 | 59.008 | 0.026 |
| `cudaStreamSynchronize` | 100 | 58.773 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 17.523 | 9.504 |
| `cudaGraphLaunch` | 63 | 16.480 | 0.001 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 14.695 | 9.504 |
| `aten::linear` | 176 | 6.616 | 54.035 |
| `Torch-Compiled Region: 2/3` | 56 | 5.788 | 0.630 |
| `aten::matmul` | 176 | 5.472 | 54.035 |
| `Torch-Compiled Region: 0/6` | 56 | 5.419 | 0.864 |
| `aten::mm` | 176 | 5.163 | 54.035 |
| `Torch-Compiled Region: 1/3` | 28 | 3.718 | 0.720 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 56 | 3.470 | 0.630 |
| `cuLaunchKernel` | 517 | 3.129 | 0.000 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 56 | 3.038 | 0.864 |

