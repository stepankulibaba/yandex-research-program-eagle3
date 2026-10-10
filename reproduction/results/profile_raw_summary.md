> Вывод `profile_eagle.py` второго прогона (с метками фаз). Колонка «GPU busy» и строки `phase: ...` в топах здесь
> неверны: метки фаз посчитались как GPU-время (исправлено в коде). Верные доли — в [profile_3a.md](profile_3a.md).

# Where the time goes: NVIDIA H200 NVL, torch 2.5.1+cu124, mt_bench x 5

EAGLE 295.0 tok/s, plain 78.6 tok/s, speed-up 3.75x, tau 6.27; one round = 21.27 ms = 1.67 plain steps (12.72 ms).

Target weights 16.1 GB: reading them once takes ~3.3 ms at 4.8 TB/s.

| one EAGLE round | ms |
|---|---|
| draft (topK_genrate) | 6.81 |
| verify (target forward over the tree) | 13.58 |
| accept (evaluate_posterior) | 0.13 |
| update without the draft | 0.22 |
| rest of the loop (Python, stop checks) | 0.13 |
| prefill per question (initialize_tree), spread over rounds | 0.51 |

| one plain token | ms |
|---|---|
| target forward | 12.74 |
| rest of the loop (Python, stop checks) | 0.11 |

| torch.profiler, 1 question | wall ms/step | GPU ms/step | GPU busy | kernel launches/step |
|---|---|---|---|---|
| eagle | 35.45 | 62.67 | 177% | 2971 |
| plain | 22.78 | 31.02 | 136% | 1820 |

Top CPU ops, eagle (ms over the question):
| op | calls | ms |
|---|---|---|
| cudaLaunchKernel | 209923 | 599.2 |
| phase: verify | 84 | 431.2 |
| aten::mm | 24650 | 260.8 |
| phase: draft | 85 | 238.8 |
| aten::mul | 38930 | 122.4 |
| aten::add | 30585 | 98.6 |
| aten::copy_ | 48098 | 93.4 |
| aten::bmm | 9520 | 91.9 |
| aten::cat | 13589 | 61.0 |
| aten::empty_strided | 31189 | 54.2 |
| cudaLaunchKernelExC | 16697 | 51.3 |
| aten::matmul | 34170 | 48.1 |

Top GPU ops, eagle (ms over the question):
| op | calls | ms |
|---|---|---|
| phase: verify | 84 | 1880.7 |
| phase: update | 84 | 1029.5 |
| phase: draft | 85 | 1009.1 |
| aten::mm | 24650 | 638.4 |
| sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1 | 6268 | 251.5 |
| sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x | 9325 | 236.4 |
| aten::copy_ | 48098 | 154.3 |
| aten::mul | 38930 | 95.9 |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_128x64_64x6_t | 6813 | 77.1 |
| aten::bmm | 9520 | 72.2 |
| aten::mean | 8245 | 70.5 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at: | 8245 | 70.5 |

Top CPU ops, plain (ms over the question):
| op | calls | ms |
|---|---|---|
| phase: target forward | 514 | 2738.7 |
| cudaLaunchKernel | 824360 | 2333.1 |
| aten::mm | 115650 | 1120.7 |
| aten::mul | 181956 | 573.3 |
| aten::bmm | 49344 | 463.5 |
| aten::copy_ | 201520 | 386.6 |
| aten::add | 115652 | 358.5 |
| cudaLaunchKernelExC | 84418 | 260.3 |
| aten::matmul | 164994 | 253.3 |
| aten::cat | 49857 | 236.9 |
| aten::empty_strided | 135182 | 229.7 |
| aten::slice | 315594 | 201.9 |

Top GPU ops, plain (ms over the question):
| op | calls | ms |
|---|---|---|
| phase: target forward | 514 | 11615.5 |
| aten::mm | 115650 | 2627.6 |
| sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1 | 33441 | 1279.9 |
| sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x | 16416 | 590.0 |
| aten::copy_ | 201520 | 545.1 |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize | 32832 | 526.2 |
| aten::mul | 181956 | 301.1 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp | 65824 | 264.6 |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, | 32832 | 226.5 |
| aten::bmm | 49344 | 200.4 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k | 83267 | 172.4 |
| aten::add | 115652 | 152.7 |
