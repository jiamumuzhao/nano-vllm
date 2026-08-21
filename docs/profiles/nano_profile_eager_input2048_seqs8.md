# Nano-vLLM Profile - eager input2048 seqs8

- Created at: `2026-07-12T08:58:38+00:00`
- Command: `/opt/anaconda3/bin/python scripts/profile_nano_long_context.py --model /root/huggingface/Qwen3-0.6B --mode eager --num-seqs 8 --input-len 2048 --output-len 64 --warmup-runs 1 --dtype float16 --max-model-len 4096 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --output-dir docs/profiles --profile-steps 12 --top-k 20`
- Prompt SHA-256: `526e32e80450f02bf23d5de00147bafdcc1308abd51e84a33c67f858c94ce9d6`

## Metrics

| Metric | Value |
| --- | ---: |
| Initialization seconds | 77.7204 |
| Prefill tokens | 2048 |
| Prefill synchronized seconds | 2.2105 |
| Prefill tok/s | 926.49 |
| Decode tokens | 504 |
| Decode synchronized seconds | 2.2099 |
| Decode tok/s | 228.07 |
| Decode step mean seconds | 0.0351 |
| Decode step median seconds | 0.0350 |
| Decode step p95 seconds | 0.0354 |
| Max CUDA allocated bytes | 3834699776 |
| Max CUDA reserved bytes | 3932160000 |

## Top CUDA Events

| Event | Count | CUDA total ms | CPU total ms |
| --- | ---: | ---: | ---: |
| `_paged_prefill_kernel` | 28 | 2155.047 | 0.000 |
| `_decode_paged_kernel` | 1764 | 311.086 | 0.000 |
| `aten::linear` | 7232 | 204.793 | 271.294 |
| `aten::matmul` | 7232 | 204.793 | 221.887 |
| `aten::mm` | 7232 | 204.793 | 208.497 |
| `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8::Params)` | 7056 | 131.111 | 0.000 |
| `turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn` | 64 | 36.582 | 0.000 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 112 | 33.639 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 19.270 | 168.108 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 19.266 | 0.000 |
| `Torch-Compiled Region: 4/1` | 64 | 10.134 | 17.305 |
| `## Call CompiledFxGraph f357sjdjyplbngsug562dtsmab6kgwiqbtaqy37wdgkjeb7mqywa ##` | 64 | 10.134 | 14.663 |
| `Torch-Compiled Region: 2/1` | 3584 | 10.083 | 362.235 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 3584 | 10.083 | 218.685 |
| `Torch-Compiled Region: 0/3` | 3584 | 9.067 | 335.771 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 3584 | 9.067 | 186.532 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 8.510 | 1.513 |
| `triton_red_fused__softmax__to_copy_argmax_clamp_min_div_exponential_ge_mul_neg_scalar_tensor_sub_unsqueeze_where_4` | 64 | 8.510 | 0.000 |
| `Torch-Compiled Region: 1/1` | 1792 | 8.024 | 235.136 |
| `## Call CompiledFxGraph fbhs6ymcy6jox7vu4d6dkickoul4rv6ilylifbsxochsg5stucq3 ##` | 1792 | 8.024 | 161.575 |

## Top CPU Events

| Event | Count | CPU total ms | CUDA total ms |
| --- | ---: | ---: | ---: |
| `aten::is_nonzero` | 28 | 2086.638 | 0.027 |
| `aten::item` | 28 | 2086.609 | 0.027 |
| `aten::_local_scalar_dense` | 28 | 2086.562 | 0.027 |
| `cudaStreamSynchronize` | 92 | 2086.297 | 0.000 |
| `Torch-Compiled Region: 2/1` | 3584 | 362.235 | 10.083 |
| `Torch-Compiled Region: 0/3` | 3584 | 335.771 | 9.067 |
| `aten::linear` | 7232 | 271.294 | 204.793 |
| `Torch-Compiled Region: 1/1` | 1792 | 235.136 | 8.024 |
| `aten::matmul` | 7232 | 221.887 | 204.793 |
| `## Call CompiledFxGraph fsjx2fzq6f66fdvmkess2wnh3ykvcsydlw7b7otbotsrfrowqajl ##` | 3584 | 218.685 | 10.083 |
| `aten::mm` | 7232 | 208.497 | 204.793 |
| `## Call CompiledFxGraph fbkmx6rszo4ias2ztqryd75uwrstrvc25ozitrkci36bxhdm2mhq ##` | 3584 | 186.532 | 9.067 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 7232 | 168.108 | 19.270 |
| `## Call CompiledFxGraph fbhs6ymcy6jox7vu4d6dkickoul4rv6ilylifbsxochsg5stucq3 ##` | 1792 | 161.575 | 8.024 |
| `Torch-Compiled Region: 3/1` | 1792 | 158.075 | 5.263 |
| `cuLaunchKernel` | 19984 | 129.182 | 0.004 |
| `TorchDynamo Cache Lookup` | 10880 | 116.665 | 0.000 |
| `aten::to` | 835 | 112.798 | 0.200 |
| `aten::_to_copy` | 450 | 111.982 | 0.200 |
| `aten::copy_` | 450 | 108.611 | 0.200 |

