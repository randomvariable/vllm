# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Extracted verbatim from ATOM atom/model_ops/moe.py.
# Source: https://github.com/ROCm/ATOM/blob/main/atom/model_ops/moe.py
#
# This file contains ONLY the gate/up interleave (GUInterleave) logic from
# ATOM's MoE implementation: the gate_mode selection, the is_guinterleave flag,
# and the preshuffle weight layout. No other MoE code is included.

"""MoE gate/up interleave (GUInterleave) logic.

ATOM's MoE supports two gate/up weight layouts:
  - GateMode.SEPARATED: traditional [gate | up] column-stacked layout
  - GateMode.INTERLEAVE: rows interleaved gate/up for better decode throughput
    on gfx1250 (and evaluated on gfx1151 / RDNA3.5 portable kernels).

The interleave mode is selected at class-init time via the env var
ATOM_MOE_GU_ITLV (see atom.utils.envs). At forward time, the gate_mode string
is passed to aiter's fused_moe / rocm_aiter_fused_moe so the kernel can consume
the interleaved layout without a runtime reshape.

gfx1151 applicability:
  The interleave layout is portable across RDNA3/3.5 (gfx1100/gfx1151). The
  per_1x32 quant path (FP8 per-1x32 block) also selects INTERLEAVE mode to
  match the preshuffled decode weight layout.
"""

from typing import Optional

import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode


# ---------------------------------------------------------------------------
# Gate/up interleave selection (from Mxfp4MoEMethod.__init__, line ~866)
# ---------------------------------------------------------------------------
# self.is_guinterleave = envs.ATOM_MOE_GU_ITLV
#
# When True, the fused_moe call receives gate_mode=GateMode.INTERLEAVE.value
# instead of the default GateMode.SEPARATED.value.


# ---------------------------------------------------------------------------
# Forward-time gate_mode dispatch (from FusedMoEMethodBase.apply, line ~1385)
# ---------------------------------------------------------------------------
def get_moe_gate_mode_interleave(is_guinterleave: bool) -> str:
    """Return the gate_mode string for fused_moe based on GUInterleave flag."""
    return (
        GateMode.INTERLEAVE.value
        if is_guinterleave
        else GateMode.SEPARATED.value
    )


# ---------------------------------------------------------------------------
# FP8 per_1x32 quant path gate_mode (from Int8MoEMethod.apply, line ~2290)
# ---------------------------------------------------------------------------
def get_moe_gate_mode_per1x32(quant_type: int) -> str:
    """Return the gate_mode string for FP8 per_1x32 preshuffled layout.

    The per_1x32 quant path uses INTERLEAVE to match the preshuffled decode
    weight layout. Other FP8 quant modes keep the historical separated layout.
    """
    return (
        GateMode.INTERLEAVE.value
        if quant_type == QuantType.per_1x32
        else GateMode.SEPARATED.value
    )


# ---------------------------------------------------------------------------
# Decode weight preshuffle (from Mxfp4MoEMethod.process_weights_after_loading,
# line ~1154)
# ---------------------------------------------------------------------------
def preshuffle_decode_weights(w13_weight: torch.Tensor, w2_weight: torch.Tensor):
    """Create zero-copy decode views of FlyDSL-shuffled weights.

    On gfx1250 (and evaluated on gfx1151), the a8w4 decode kernel expects
    weights in an interleaved row layout. moe_weight_decode_view returns a
    view that shares storage with the original weight tensor — no second copy.

    Returns (w13_weight_preshuffled, w2_weight_preshuffled).
    """
    from aiter.ops.triton.utils.shuffle import moe_weight_decode_view

    w13_weight_preshuffled = moe_weight_decode_view(w13_weight.data)
    w2_weight_preshuffled = moe_weight_decode_view(w2_weight.data)
    return w13_weight_preshuffled, w2_weight_preshuffled


# ---------------------------------------------------------------------------
# fused_moe call with gate_mode (from FusedMoEMethodBase.apply, line ~1393)
# ---------------------------------------------------------------------------
def fused_moe_with_gate_mode(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    is_guinterleave: bool,
    swiglu_limit: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Call aiter fused_moe with the correct gate_mode for GUInterleave."""
    moe_extra_args = {
        "gate_mode": get_moe_gate_mode_interleave(is_guinterleave),
        "swiglu_limit": swiglu_limit,
    }
    return fused_moe(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        **kwargs,
        **moe_extra_args,
    )
