# Nano-vLLM Profile - graph input2048 seqs8

- Created at: `2026-07-12T09:00:45+00:00`
- Command: `/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode graph --num-seqs 8 --input-len 2048 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20`
- Prompt SHA-256: `526e32e80450f02bf23d5de00147bafdcc1308abd51e84a33c67f858c94ce9d6`

## Metrics

| Metric | Value |
| --- | ---: |
| Initialization seconds | 79.1273 |
| Prefill tokens | 2048 |
| Prefill synchronized seconds | 2.2530 |
| Prefill tok/s | 909.02 |
| Decode tokens | 504 |
| Decode synchronized seconds | 0.6353 |
| Decode tok/s | 793.32 |
| Decode step mean seconds | 0.0101 |
| Decode step median seconds | 0.0097 |
| Decode step p95 seconds | 0.0121 |
| Max CUDA allocated bytes | 3843239424 |
| Max CUDA reserved bytes | 3957325824 |

## Top CUDA Events

| Event | Count | CUDA total ms | CPU total ms |
| --- | ---: | ---: | ---: |
| `_paged_prefill_kernel` | 28 | 2191.887 | 0.000 |
| `_decode_paged_kernel` | 1764 | 360.211 | 0.000 |
| `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8::Params)` | 7056 | 142.677 | 0.000 |
| `aten::linear` | 176 | 78.047 | 6.782 |
| `aten::matmul` | 176 | 78.047 | 5.643 |
| `aten::mm` | 176 | 78.047 | 5.325 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 112 | 39.285 | 0.000 |
| `turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn` | 64 | 38.762 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 23.065 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 11.758 | 17.398 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 11.758 | 14.632 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 9.975 | 1.498 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 9.975 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_0` | 1792 | 5.918 | 0.000 |
| `triton_poi_fused_mul_silu_split_0` | 1792 | 5.882 | 0.000 |
| `store_kvcache_kernel` | 1792 | 5.111 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_1` | 1792 | 4.695 | 0.000 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, __half, false, __half, __half, __half, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __half const*, __half const*, __half*, __half*, float const*, float const*, __half const*, __half const*, __half*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | 1764 | 4.544 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 113 | 3.551 | 2.705 |
| `Torch-Compiled Region: 3/3` | 28 | 2.254 | 2.474 |

## Top CPU Events

| Event | Count | CPU total ms | CUDA total ms |
| --- | ---: | ---: | ---: |
| `aten::is_nonzero` | 28 | 2128.775 | 0.027 |
| `aten::item` | 28 | 2128.746 | 0.027 |
| `aten::_local_scalar_dense` | 28 | 2128.701 | 0.027 |
| `cudaStreamSynchronize` | 100 | 2128.460 | 0.000 |
| `aten::to` | 717 | 651.674 | 0.176 |
| `aten::copy_` | 655 | 651.496 | 0.494 |
| `aten::_to_copy` | 395 | 650.974 | 0.176 |
| `cudaMemcpyAsync` | 682 | 647.850 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 17.398 | 11.758 |
| `cudaGraphLaunch` | 63 | 16.000 | 0.000 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 14.632 | 11.758 |
| `aten::linear` | 176 | 6.782 | 78.047 |
| `Torch-Compiled Region: 2/3` | 56 | 5.718 | 1.735 |
| `aten::matmul` | 176 | 5.643 | 78.047 |
| `aten::mm` | 176 | 5.325 | 78.047 |
| `Torch-Compiled Region: 0/6` | 56 | 5.304 | 1.799 |
| `Torch-Compiled Region: 1/3` | 28 | 3.771 | 1.427 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 56 | 3.497 | 1.735 |
| `cuLaunchKernel` | 517 | 3.056 | 0.000 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 56 | 2.999 | 1.799 |

