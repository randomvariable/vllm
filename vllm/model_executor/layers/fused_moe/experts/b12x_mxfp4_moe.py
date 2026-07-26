# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X modular fused-MoE backend for DeepSeek V4 native MXFP4 weights."""

import importlib
import importlib.metadata
import inspect
from typing import Any, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _run_b12x_moe_fp4(
    *,
    a: torch.Tensor,
    experts: Any,
    scratch_plan: Any,
    scratch: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_scales_static: bool,
    unit_scale_contract: bool,
) -> None:
    """Call b12x MoE with preplanned, expert-owned scratch."""
    from b12x.integration.tp_moe import (  # type: ignore[import-not-found]
        b12x_moe_fp4,
    )

    binding = scratch_plan.bind(
        scratch=scratch,
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=input_scales_static,
        unit_scale_contract=unit_scale_contract,
    )
    b12x_moe_fp4(binding=binding)


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation == MoEActivation.RELU2:
        return "relu2"
    return activation.value


def _plan_b12x_fp4_moe_weights(**kwargs):
    from b12x.integration import (  # type: ignore[import-not-found]
        plan_b12x_fp4_moe_weights,
    )

    return plan_b12x_fp4_moe_weights(**kwargs)


def _prepare_b12x_fp4_moe_weights(**kwargs):
    from b12x.integration import (  # type: ignore[import-not-found]
        prepare_b12x_fp4_moe_weights,
    )

    return prepare_b12x_fp4_moe_weights(**kwargs)


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    param_name: str,
) -> torch.Tensor | None:
    param = getattr(layer, param_name, None)
    if not isinstance(param, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=param.dtype, device=param.device)
    replace_parameter(layer, param_name, empty)
    return param


def _set_quant_config_weight_scale(
    quant_config: FusedMoEQuantConfig,
    weight_name: str,
    scale: torch.Tensor,
) -> None:
    desc = getattr(quant_config, weight_name, None)
    if desc is not None and hasattr(desc, "scale"):
        desc.scale = scale
        return

    public_name = "w1_scale" if weight_name == "_w1" else "w2_scale"
    if hasattr(quant_config, public_name):
        setattr(quant_config, public_name, scale)


def _maybe_release_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or _is_current_stream_capturing():
        return
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        accelerator.empty_cache()
    else:
        torch.cuda.empty_cache()


def _raise_if_capture_copy_required(tensor: torch.Tensor, description: str) -> None:
    if tensor.device.type != "cuda" or not _is_current_stream_capturing():
        return
    raise RuntimeError(
        f"B12X MoE {description} would allocate during CUDA graph capture"
    )


def _is_current_stream_capturing() -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    is_capturing = getattr(cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _normalize_b12x_moe_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if topk_ids.dtype != torch.int32:
        _raise_if_capture_copy_required(topk_ids, "topk_ids dtype normalization")
        topk_ids = topk_ids.to(torch.int32)
    if not topk_ids.is_contiguous():
        _raise_if_capture_copy_required(topk_ids, "topk_ids contiguity normalization")
        topk_ids = topk_ids.contiguous()
    return topk_ids


def _normalize_b12x_moe_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype != torch.float32:
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights dtype normalization",
        )
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights contiguity normalization",
        )
        topk_weights = topk_weights.contiguous()
    return topk_weights


def has_b12x_moe() -> bool:
    """Return whether b12x 0.30.2 exposes the API used by this backend."""
    try:
        if importlib.metadata.version("b12x") != "0.30.2":
            return False
        integration = importlib.import_module("b12x.integration")
        tp_moe = importlib.import_module("b12x.integration.tp_moe")
        expected = {
            integration.plan_b12x_fp4_moe_weights: {
                "quant_modes",
                "num_experts",
                "hidden_size",
                "intermediate_size",
            },
            integration.prepare_b12x_fp4_moe_weights: {"plan"},
            integration.B12XFP4ExpertWeights: {
                "plan",
                "w1_fp4",
                "w1_blockscale",
                "w2_fp4",
                "w2_blockscale",
            },
            tp_moe.TPMoEScratchCaps: {"weight_plan", "max_tokens"},
            tp_moe.TPMoEScratchPlan.bind: {"experts", "scratch"},
        }
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError):
        return False
    return hasattr(integration, "B12XFP4ExpertWeights") and all(
        required <= set(inspect.signature(obj).parameters)
        for obj, required in expected.items()
    )


class B12xExperts(mk.FusedMoEExpertsModular):
    """Native DeepSeek V4 MXFP4 MoE backend powered by b12x kernels."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype == "mxfp4", (
            "B12xExperts only supports native MXFP4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self._experts: Any | None = None
        self._weight_plan: Any | None = None
        self._scratch_plan: Any | None = None
        self._scratch: tuple[torch.Tensor, ...] = ()
        self._released_w4a16_source_scales = False
        self._unit_scale_by_device: dict[torch.device, torch.Tensor] = {}

    def _source_format(self) -> str:
        return "fp4_e8m0_k32"

    def _w13_layout(self) -> str:
        # vLLM DSV4 loading stores fused W13 as [w1/gate, w3/up], which is the
        # row order consumed by b12x for the runtime SwiGLU path.
        return "w31"

    def _unit_expert_scale(
        self, device: torch.device, num_experts: int
    ) -> torch.Tensor:
        scale = self._unit_scale_by_device.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scale_by_device[device] = scale
        return scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare b12x-owned W4A16 weights and release one-way sources."""
        if self._experts is not None:
            raise RuntimeError(
                "B12X weights have already been prepared; rebuild the engine to "
                "load new weights"
            )

        w13_weight = cast(torch.Tensor, layer.w13_weight)
        w2_weight = cast(torch.Tensor, layer.w2_weight)
        device = w13_weight.device
        moe_config = getattr(self, "moe_config", None)
        params_dtype = getattr(moe_config, "in_dtype", torch.bfloat16)
        activation = getattr(layer, "activation", None)
        if activation is None:
            activation = getattr(moe_config, "activation", MoEActivation.SILU)
        activation = cast(MoEActivation, activation)

        self._get_or_prepare_fp4_moe_weights(
            w1=w13_weight,
            w2=w2_weight,
            activation=activation,
            params_dtype=params_dtype,
        )
        self._assert_owner_aliases_source(w13_weight, w2_weight)
        self._release_w4a16_source_scales(layer)
        self._release_w4a16_source_weights(layer)
        _maybe_release_cuda_cache(device)

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size <= 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method == RoutingMethodType.DeepseekV4

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _get_or_prepare_fp4_moe_weights(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ):
        if self._experts is not None:
            return self._experts

        if self._released_w4a16_source_scales:
            raise RuntimeError(
                "B12X W4A16 source block scales were already released; "
                f"cannot prepare FP4 MoE weights for dtype {params_dtype}."
            )

        if w1.device.type == "cuda" and _is_current_stream_capturing():
            raise RuntimeError(
                "B12X FP4 MoE weights were not prepared before CUDA "
                f"graph capture for dtype {params_dtype}."
            )
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )

        unit_scale = self._unit_expert_scale(w1.device, int(w1.shape[0]))
        self._weight_plan = _plan_b12x_fp4_moe_weights(
            quant_modes="w4a16",
            source_format=self._source_format(),
            w13_layout=self._w13_layout(),
            activation=_b12x_activation_name(activation),
            params_dtype=params_dtype,
            num_experts=int(w1.shape[0]),
            hidden_size=int(self.moe_config.hidden_dim),
            intermediate_size=int(self.moe_config.intermediate_size_per_partition),
        )
        self._experts = _prepare_b12x_fp4_moe_weights(
            plan=self._weight_plan,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_global_scale=unit_scale,
            a1_gscale=unit_scale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_global_scale=unit_scale,
            a2_gscale=unit_scale,
            params_dtype=params_dtype,
        )
        self._allocate_scratch(w1.device)
        return self._experts

    def _lookup_prepared_w4a16(self) -> Any | None:
        return self._experts

    def _allocate_scratch(self, device: torch.device) -> None:
        from b12x.integration.tp_moe import (  # type: ignore[import-not-found]
            TPMoEScratchCaps,
            plan_tp_moe_scratch,
        )

        max_tokens = max(
            1,
            int(self.moe_config.max_num_tokens) * int(self.moe_config.dp_size),
            int(self.moe_config.max_capture_size),
        )
        scratch_plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                max_tokens=max_tokens,
                num_topk=int(self.moe_config.experts_per_token),
                device=device,
                weight_plan=self._weight_plan,
                quant_mode="w4a16",
                core_token_counts=(max_tokens,),
                route_num_experts=0,
                apply_router_weight_on_input=False,
                swiglu_limit=getattr(self.quant_config, "gemm1_clamp_limit", None),
                frozen=True,
            )
        )
        self._scratch_plan = scratch_plan
        self._scratch = tuple(
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in scratch_plan.shapes_and_dtypes()
        )

    def _assert_owner_aliases_source(
        self, w1: torch.Tensor, w2: torch.Tensor
    ) -> None:
        assert self._experts is not None
        assert self.w1_scale is not None and self.w2_scale is not None
        w1_scale = self.w1_scale
        w2_scale = self.w2_scale
        aliases = (
            self._experts.w1_fp4.untyped_storage().data_ptr()
            == w1.untyped_storage().data_ptr()
            and self._experts.w2_fp4.untyped_storage().data_ptr()
            == w2.untyped_storage().data_ptr()
            and self._experts.w1_blockscale.untyped_storage().data_ptr()
            == w1_scale.untyped_storage().data_ptr()
            and self._experts.w2_blockscale.untyped_storage().data_ptr()
            == w2_scale.untyped_storage().data_ptr()
        )
        if not aliases:
            raise RuntimeError(
                "B12X prepared owner does not alias canonical source storage; "
                "refusing to release source parameters"
            )

    def _release_w4a16_source_scales(self, layer: torch.nn.Module) -> None:
        if self._released_w4a16_source_scales:
            return

        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w1", w1_scale)
        if w2_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w2", w2_scale)

        self._released_w4a16_source_scales = True

    def _release_w4a16_source_weights(self, layer: torch.nn.Module) -> None:
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() != 0 and w2.numel() != 0:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        experts = self._lookup_prepared_w4a16()
        if experts is None:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            m = a1.size(0)
        else:
            assert a1.dim() == 3
            m = a1.size(1)

        intermediate_size = int(experts.plan.intermediate_size)
        n = intermediate_size * 2
        return (
            int(experts.plan.num_experts),
            m,
            n,
            a1.size(-1),
            topk_ids.size(1),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Scratch has heterogeneous dtypes and is persistently expert-owned.
        return (1,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        if self._experts is None or self._scratch_plan is None or not self._scratch:
            raise RuntimeError(
                "B12X MoE weights and scratch were not prepared before execution"
            )

        if expert_map is not None:
            raise RuntimeError(
                "B12X MoE does not support expert_map with the current b12x_moe_fp4 API"
            )
        if apply_router_weight_on_input:
            raise RuntimeError(
                "B12X MoE scratch was planned without input router weighting"
            )

        topk_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        topk_weights = _normalize_b12x_moe_topk_weights(topk_weights)

        _run_b12x_moe_fp4(
            a=hidden_states,
            experts=self._experts,
            scratch_plan=self._scratch_plan,
            scratch=self._scratch,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
            unit_scale_contract=True,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
