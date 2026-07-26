# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions derived from ROCm/ATOM PR #1337 (MIT License, Advanced Micro Devices).
# Dispatches to aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant.fused_moe_int8_smoothquant
# (pure Triton, portable across archs, validated on gfx1151 RDNA3.5).
# Requires aiter with fused_moe_int8_smoothquant (aiter PR #3917).
"""Online INT8 W8A8 MoE method for ROCm (gfx1151 / RDNA3.5).

Quantizes BF16 checkpoint weights to int8 on-the-fly at load time with
per-channel weight scales and per-token activation quantization. This is
"online" INT8: the checkpoint stays BF16, weights are dynamically int8-quantized
per output channel, and activations are dynamically int8-quantized per token.

Value on gfx1151 (bandwidth-bound APU): ~36% decode speedup for 27B models,
35B-A3B 41 tok/s @8K context, gsm8k 0.84 (BF16-equivalent quality).

Gated by VLLM_ROCM_USE_AITER_ONLINE_INT8_MOE (default OFF).
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

# Module-level sentinel cache for aiter kernel (populated once).
_AITER_KERNEL = None


def _get_aiter_kernel():
    """Lazy-load aiter's fused_moe_int8_smoothquant kernel (module-level cache)."""
    global _AITER_KERNEL
    if _AITER_KERNEL is None:
        try:
            from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
                fused_moe_int8_smoothquant,
            )

            _AITER_KERNEL = fused_moe_int8_smoothquant
        except ImportError as e:
            raise ImportError(
                "Online INT8 MoE requires aiter with "
                "fused_moe_int8_smoothquant (aiter PR #3917). "
                "Install or update aiter. Error: "
            ) from e
    return _AITER_KERNEL


def _is_rocm_gfx1151() -> bool:
    """Check if current device is gfx1151 (Strix Halo / RDNA3.5)."""
    try:
        from vllm.platforms.rocm import on_gfx1151

        return on_gfx1151()
    except (ImportError, AttributeError):
        return False


def _is_online_int8_moe_enabled() -> bool:
    """Check if online INT8 MoE is enabled via env var."""
    return envs.VLLM_ROCM_USE_AITER_ONLINE_INT8_MOE


def _interleave_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up rows for fused SiLU-gated expert computation.

    ATOM's permutation trick: transpose [E, out, in] -> [E, in, out], then
    interleave gate/up columns so fused _swiglu splits adjacent pairs.
    Input w13 shape: [E, 2*intermediate, hidden] (gate then up concatenated).
    Output shape: [E, hidden, 2*intermediate] with interleaved (g0,u0,g1,u1,...).

    Args:
        w13: Gate+up weight tensor [E, 2*I, H] where I=intermediate, H=hidden.

    Returns:
        Interleaved weight tensor [E, H, 2*I] with (g0,u0,g1,u1,...) layout.
    """
    num_experts = w13.shape[0]
    two_intermediate = w13.shape[1]
    hidden = w13.shape[2]
    intermediate = two_intermediate // 2

    # Split gate and up: [E, I, H] each
    gate = w13[:, :intermediate, :]
    up = w13[:, intermediate:, :]

    # Transpose to [E, H, I] each
    gate_t = gate.transpose(-2, -1)
    up_t = up.transpose(-2, -1)

    # Interleave: stack along new dim, then reshape to [E, H, 2*I]
    # Result: (g0,u0,g1,u1,...,g_{I-1},u_{I-1})
    interleaved = torch.stack([gate_t, up_t], dim=-1)
    interleaved = interleaved.reshape(num_experts, hidden, two_intermediate)

    return interleaved.contiguous()


def _interleave_gate_up_scales(w13_scale: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up scales following the same permutation as weights.

    Input w13_scale shape: [E, 2*I, 1] (per-output-channel scales).
    Output shape: [E, 2*I, 1] with interleaved (g0,u0,g1,u1,...).

    Args:
        w13_scale: Gate+up scale tensor [E, 2*I, 1].

    Returns:
        Interleaved scale tensor [E, 2*I, 1].
    """
    num_experts = w13_scale.shape[0]
    two_intermediate = w13_scale.shape[1]
    intermediate = two_intermediate // 2

    # Squeeze last dim: [E, 2*I]
    scales = w13_scale.squeeze(-1)

    # Split gate and up scales: [E, I] each
    gate_scales = scales[:, :intermediate]
    up_scales = scales[:, intermediate:]

    # Interleave: stack along new dim, then reshape to [E, 2*I]
    interleaved = torch.stack([gate_scales, up_scales], dim=-1)
    interleaved = interleaved.reshape(num_experts, two_intermediate)

    return interleaved.contiguous().unsqueeze(-1)


class OnlineInt8MoEMethod(FusedMoEMethodBase):
    """Online INT8 W8A8 MoE method using aiter's Triton kernel.

    Quantizes BF16 checkpoint weights to int8 on-the-fly at load time.
    Activations are dynamically int8-quantized per token. Dispatches to
    aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant.fused_moe_int8_smoothquant.

    SiLU-gated experts only (activation=Silu).

    Requires:
        - ROCm platform (gfx1151 validated, portable across archs)
        - aiter with fused_moe_int8_smoothquant (aiter PR #3917)
        - VLLM_ROCM_USE_AITER_ONLINE_INT8_MOE=1
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create int8 weight parameters and per-channel fp32 scales.

        Weights are stored as int8 [E, out, in] with per-output-channel fp32
        scales. The checkpoint is BF16; weights will be quantized in
        process_weights_after_loading.
        """
        # w13: gate+up fused [E, 2*I, H]
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # w2: down projection [E, H, I]
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-output-channel fp32 scales for w13 and w2.
        # w13_scale: [E, 2*I, 1] (will be interleaved to [E, 2*I, 1] after loading)
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        # w2_scale: [E, H, 1] (per-output-channel)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Add PER-CHANNEL quantization marker for weight loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # No static input scales (activations are dynamically quantized per token).
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transpose and interleave weights for fused SiLU-gated computation.

        Transposes [E, out, in] -> [E, in, out] and interleaves gate/up so the
        aiter kernel's fused _swiglu splits adjacent pairs (g0,u0,g1,u1,...).
        Scales follow the same permutation.

        For online INT8, the checkpoint loader has already quantized BF16 weights
        to int8 and computed per-channel scales. We just rearrange them here.
        """
        # Interleave gate/up in w13: [E, 2*I, H] -> [E, H, 2*I] with (g,u,g,u,...)
        w13_interleaved = _interleave_gate_up(layer.w13_weight.data)
        layer.w13_weight.data = w13_interleaved

        # Interleave w13 scales: [E, 2*I, 1] -> [E, 2*I, 1] with (g,u,g,u,...)
        w13_scale_interleaved = _interleave_gate_up_scales(
            layer.w13_weight_scale.data
        )
        layer.w13_weight_scale.data = w13_scale_interleaved

        # Transpose w2: [E, H, I] -> [E, I, H]
        layer.w2_weight.data = layer.w2_weight.data.transpose(-2, -1).contiguous()

        # Squeeze w2_scale last dim: [E, H, 1] -> [E, H]
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.squeeze(-1)

        logger.info_once(
            "OnlineInt8MoEMethod: weights transposed and interleaved for "
            "aiter fused_moe_int8_smoothquant kernel."
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        """Return quant config for online INT8.

        Returns None since we don't use the modular kernel framework;
        we dispatch directly to aiter's fused_moe_int8_smoothquant.
        """
        return None

    def _call_aiter_kernel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Call aiter's fused_moe_int8_smoothquant kernel.

        Args:
            layer: RoutedExperts instance with int8 weights and scales.
            x: Input hidden states [M, H] in bf16/fp16.
            gating_output: Router logits [M, E] (raw, pre-softmax).
            topk: Number of experts to select per token.

        Returns:
            Output tensor [M, H] after MoE computation.
        """
        kernel_fn = _get_aiter_kernel()

        # aiter's fused_moe_int8_smoothquant signature:
        #   hidden_states: [M, H] bf16/fp16
        #   w13: [E, H, 2*I] int8 (kernel layout K=H, N=2*I, interleaved g,u)
        #   w2: [E, I, H] int8 (kernel layout K=I, N=H)
        #   w13_scale: [E, 2*I] fp32 per-output-channel (interleaved)
        #   w2_scale: [E, H] fp32 per-output-channel
        #   gating_output: [M, E] routed-expert logits (kernel does routing)
        #   topk: int
        #   renormalize: bool
        #   dtype: torch.dtype
        out_dtype = x.dtype
        w13_scale = layer.w13_weight_scale.squeeze(-1)

        output = kernel_fn(
            hidden_states=x,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=w13_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=gating_output,
            topk=topk,
            renormalize=True,
            dtype=out_dtype,
        )
        return output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch to aiter's fused_moe_int8_smoothquant kernel.

        The aiter kernel expects raw router logits [M, E] and does its own
        routing internally. Since apply() receives pre-computed topk_weights
        and topk_ids, we reconstruct pseudo-logits via log(topk_weights) and
        scatter them into a [M, E] matrix. The kernel's internal softmax will
        recover the correct routing weights.

        For production use, prefer apply_monolithic() which receives router
        logits directly (no reconstruction needed).

        Args:
            layer: RoutedExperts instance with int8 weights and scales.
            x: Input hidden states [M, H] in bf16/fp16.
            topk_weights: Expert routing weights [M, top_k] (post-softmax).
            topk_ids: Selected expert IDs [M, top_k].
            shared_experts: Shared experts module (not fused in this draft).
            shared_experts_input: Shared experts input (not used here).

        Returns:
            Output tensor [M, H] after MoE computation.
        """
        raise NotImplementedError(
            "OnlineInt8MoEMethod.apply() pre-computed topk_weights/topk_ids "
            "path is not supported. Use apply_monolithic() with raw "
            "router_logits instead."
        )

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Monolithic apply with internal routing.

        This is the preferred path: receives raw router_logits [M, E] and
        passes them directly to the aiter kernel, which does its own routing.

        Args:
            layer: RoutedExperts instance with int8 weights and scales.
            x: Input hidden states [M, H].
            router_logits: Router logits [M, E] (raw, pre-softmax).
            input_ids: Input token IDs (not used).

        Returns:
            Output tensor [M, H].
        """
        # Determine topk from moe config.
        top_k = self.moe.top_k if hasattr(self.moe, "top_k") else 1
        return self._call_aiter_kernel(layer, x, router_logits, top_k)
