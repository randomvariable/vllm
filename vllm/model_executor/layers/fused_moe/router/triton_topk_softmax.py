# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton top-k softmax routing kernel for ROCm.

Provides a device-native fallback for MoE expert routing (softmax over the
router logits followed by top-k selection) on ROCm GPUs where neither AITER
nor the CUDA-only ``_moe_C.topk_softmax`` op is available — notably gfx1151
(Strix Halo), where AITER is disabled (no FP8 tensor cores).

Routing is a tiny fraction of MoE latency (the expert GEMM dominates), so this
kernel is sized for correctness and occupancy on a 40-CU APU rather than peak
throughput. The interface mirrors ``vllm_topk_softmax`` in fused_topk_router.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_softmax_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    num_experts,
    renormalize,
    stride_gating,
    stride_topk,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_experts

    logits = tl.load(
        gating_ptr + token * stride_gating + offs, mask=mask, other=-float("inf")
    ).to(tl.float32)

    # Softmax over the expert dimension (numerically stabilized).
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    neg_inf = float("-inf")
    working = probs
    w_sum = 0.0

    # Top-k via iterative masked argmax (TOPK is small and compile-time).
    for k in tl.static_range(TOPK):
        idx = tl.argmax(working, axis=0)
        w = tl.max(working, axis=0)
        tl.store(topk_weights_ptr + token * stride_topk + k, w)
        tl.store(topk_indices_ptr + token * stride_topk + k, idx)
        tl.store(token_expert_indices_ptr + token * stride_topk + k, idx)
        w_sum += w
        # Mask out the selected expert so it is not picked again.
        working = tl.where(offs == idx, neg_inf, working)

    # Renormalize the selected weights to sum to 1.
    if renormalize:
        inv = 1.0 / w_sum
        for k in tl.static_range(TOPK):
            w = tl.load(topk_weights_ptr + token * stride_topk + k)
            tl.store(topk_weights_ptr + token * stride_topk + k, w * inv)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Softmax-then-top-k expert routing backed by a Triton kernel.

    Args:
        topk_weights: [num_tokens, topk] fp32 output for expert weights.
        topk_indices: [num_tokens, topk] int32 output for expert ids.
        token_expert_indices: [num_tokens, topk] int32 output (set to expert ids).
        gating_output: [num_tokens, num_experts] router logits (fp32/fp16/bf16).
        renormalize: renormalize the selected weights to sum to 1.

    Returns:
        (topk_weights, topk_indices)
    """
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]
    block_size = _next_pow2(num_experts)

    _topk_softmax_kernel[(num_tokens,)](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_experts,
        1 if renormalize else 0,
        gating_output.stride(0),
        topk_weights.stride(0),
        TOPK=topk,
        BLOCK_SIZE=block_size,
    )
    return topk_weights, topk_indices
