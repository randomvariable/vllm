# ATOM Reference Kernels for gfx1151 (Strix Halo / RDNA3.5)

This directory contains **reference code** from [ROCm/ATOM](https://github.com/ROCm/ATOM),
evaluated for portability to **gfx1151** (AMD Strix Halo / RDNA3.5) in the
`homelabs-main` vLLM fork.

## License & Attribution

All files in this directory are:
- **License**: MIT (`SPDX-License-Identifier: MIT`)
- **Copyright**: Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
- **Source**: https://github.com/ROCm/ATOM
- **NOT a runtime dependency** — these files are reference material for evaluation and
  future integration. They are not imported by vLLM at runtime.
  - **Exception:** `attention/attention_gdn.py` is a verbatim copy that retains ATOM's
    original import structure (`from atom.*`) and therefore requires ATOM installed to
    import standalone. The other reference files are self-contained. Phase 2 integration
    replaces these imports with vLLM-native equivalents.

## Purpose

These kernels represent gfx1151-portable optimizations from ATOM that target:
- **GDN (Gated Delta Network) FLA fusions** — prefill hot path for Qwen3.6
- **QK-norm + RoPE + KV-cache-write** fusions
- **GDN full-graph decode compaction** — correctness fix for CUDA graph padded batches
- **MoE gate/up interleave** — decode throughput optimization
- **DSpark confidence scheduling** — spec decode quality/perf tradeoff
- **Spec decode micro-kernels** — verification scheduler, lm_head argmax
- **Online INT8 W8A8** — MoE quantization paths

FP8/FP4/MLA/CDNA-only ATOM code was **deliberately excluded**. Only kernels with
clear gfx1151 (RDNA3.5) applicability are included.

## File Index

| File | ATOM Source Path | What it optimizes | Target story | gfx1151 applicability |
|------|------------------|-------------------|--------------|----------------------|
| `fla/chunk_fused.py` | `atom/model_ops/fla_ops/chunk_fused.py` | FLA chunk fusion for GDN prefill | 2.1 GDN FLA fusions | Portable Triton kernel; gfx1151 via standard Triton AMD backend |
| `fla/fused_cumsum_kkt.py` | `atom/model_ops/fla_ops/fused_cumsum_kkt.py` | Fused cumulative sum for KKT solver | 2.1 GDN FLA fusions | Portable Triton; reduces kernel launch overhead |
| `fla/fused_merge_recompute.py` | `atom/model_ops/fla_ops/fused_merge_recompute.py` | Fused merge + recompute for FLA segments | 2.1 GDN FLA fusions | Portable Triton; memory-bound fusion |
| `attention/triton_fused_qkv_norm_rope_cache.py` | `atom/model_ops/triton_fused_qkv_norm_rope_cache.py` | Fused QK-norm + RoPE + KV-cache write | 2.2 QK-norm+RoPE+KV-cache-write | Portable Triton; reduces memory traffic for decode |
| `attention/attention_gdn.py` | `atom/model_ops/attention_gdn.py` | GDN attention kernel (full) | 2.1 GDN FLA fusions | Evaluated for gfx1151; may need tuning for RDNA3.5 wavefront |
| `moe/moe_gu_interleave.py` | `atom/model_ops/moe.py` (lines 866, 1154, 1385-1392, 2290-2298) | MoE gate/up interleave for decode | 2.4 MoE gate/up interleave | Portable via aiter; gfx1250 primary, gfx1151 evaluated |
| `spec_decode/dspark_scheduler.py` | `atom/spec_decode/dspark_scheduler.py` | DSpark confidence-based scheduling | 3.1 DSpark confidence scheduling | Architecture-agnostic scheduler logic |
| `spec_decode/verify_scheduler.py` | `atom/spec_decode/verify_scheduler.py` | Spec decode verification scheduler | 3.3 spec-decode micro-kernels | Architecture-agnostic scheduler logic |
| `kernels/lm_head_argmax.py` | `atom/model_ops/lm_head_argmax.py` | Fused lm_head + argmax for spec decode | 3.3 spec-decode micro-kernels | Portable Triton; reduces launch overhead |
| `kernels/fused_aux_rmsnorm.py` | `atom/model_ops/fused_aux_rmsnorm.py` | Fused auxiliary RMSNorm | 3.3 spec-decode micro-kernels | Portable Triton; memory-bound fusion |
| `kernels/swiglu_oai.py` | `atom/model_ops/swiglu_oai.py` | OpenAI-style SwiGLU activation | 3.3 spec-decode micro-kernels | Portable Triton; activation fusion |
| `plugin/gdn_backend_compaction.py` | `atom/plugin/vllm/gdn_backend.py` (lines 31-127) | GDN full-graph decode metadata compaction | 2.3 GDN full-graph decode compaction | Architecture-agnostic; correctness fix for CUDA graph padded batches |

## Target Stories

- **2.1 GDN FLA fusions (Qwen3.6 prefill hot path)**: Chunk-fused FLA ops, fused cumsum KKT,
  fused merge recompute, and the full GDN attention kernel. These target the prefill phase
  of Qwen3.6 models using Gated Delta Networks.
- **2.2 QK-norm + RoPE + KV-cache-write**: Fused Triton kernel that combines query/key
  normalization, rotary position embedding, and KV-cache write into a single kernel launch.
- **2.3 GDN full-graph decode compaction**: Correctness fix for CUDA graph execution —
  strips padded decode rows from metadata so state indexing remains valid. Critical when
  cudagraph batch size exceeds actual decode request count.
- **2.4 MoE gate/up interleave**: MoE weight layout optimization for decode throughput.
  The `GateMode.INTERLEAVE` path interleaves gate/up rows for better memory access patterns
  on RDNA3/3.5 GPUs.
- **3.1 DSpark confidence scheduling**: Confidence-based token acceptance for speculative
  decoding. Adjusts draft/verify tradeoff based on model confidence.
- **3.2 MTP-on-MoE-drafter fix**: Model fix in `atom/models/qwen3_5_mtp.py` — NOT copied
  to this directory. Reference only: the fix ensures Multi-Token Prediction works correctly
  when the drafter is an MoE model. See ATOM source for details.
- **3.3 Spec decode micro-kernels**: Fused lm_head+argmax, fused auxiliary RMSNorm,
  SwiGLU activation, and verification scheduler. Small kernels that reduce launch overhead
  in the spec decode loop.
- **4.1 Online INT8 W8A8**: INT8 weight-only quantization for MoE layers. The `Int8MoEMethod`
  class in `atom/model_ops/moe.py` (not copied here) implements the per-tensor and per-1x32
  quantization paths. The `per_1x32` path uses `GateMode.INTERLEAVE` to match the preshuffled
  decode weight layout. Reference the ATOM source for the full implementation.

## Integration Status

These files are **reference only**. Integration into vLLM will proceed in phases:
- Phase 1: Evaluate gfx1151 portability and performance
- Phase 2: Adapt kernels for vLLM's build system and dependency model
- Phase 3: Wire into vLLM's model/attention/spec-decode paths

No runtime imports from this directory exist in the vLLM codebase.
