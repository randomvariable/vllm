# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X modular fused-MoE backend for native MXFP4 and NVFP4 weights."""

import importlib
import importlib.metadata
import inspect
from typing import Any, cast

import torch
from packaging.version import InvalidVersion, Version

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
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

# b12x 1.1.0 renamed the package from sparkinfer and moved the planned MoE
# API from b12x.integration.tp_moe to b12x.moe.fused_moe.
B12X_VERSION = "1.1.0"

# Source checkpoint layout and b12x quant mode, keyed by vLLM weight dtype.
_B12X_SOURCE_FORMATS = {
    "mxfp4": "fp4_e8m0_k32",
    "nvfp4": "modelopt_nvfp4",
}
_B12X_QUANT_MODES = {
    "mxfp4": "w4a16",
    "nvfp4": "nvfp4",
}


def _run_moe(
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
    """Call b12x MoE with preplanned, shared-workspace-owned scratch."""
    from b12x.moe.fused_moe import run  # type: ignore[import-not-found]

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
    run(binding=binding)


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation == MoEActivation.RELU2:
        return "relu2"
    return activation.value


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
    torch.accelerator.empty_cache()


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


def _normalize_modelopt_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    """Collapse a ModelOpt per-expert activation scale to one value per expert.

    ModelOpt checkpoints may store the w13 activation scale as ``[E, 2]`` (one
    per fused gate/up half). b12x expects ``[E]``; the halves are equal by
    construction, so the first column is taken.

    Args:
        scale: Activation global scale, ``[E]`` or ``[E, 1|2]``.

    Returns:
        A contiguous ``[E]`` tensor.

    Raises:
        ValueError: When a 2-D scale has an unexpected second dimension.
    """
    if scale.dim() == 2:
        if scale.size(1) not in (1, 2):
            raise ValueError(
                "expected ModelOpt expert scale second dimension to be 1 or 2, "
                f"got {tuple(scale.shape)}"
            )
        scale = scale[:, 0]
    return scale.contiguous()


def has_b12x_moe() -> bool:
    """Return whether b12x exposes the planned MoE API used by this backend.

    Returns:
        False when b12x is not installed at all, which leaves other MoE
        backends free to claim the layer.

    Raises:
        RuntimeError: When b12x is installed but too old to expose the
            ``b12x.moe.fused_moe`` planned API. Downgrading to another
            backend would silently trade the SM120/SM121 kernels for a
            slower path, so an incompatible install fails loudly instead.
    """
    try:
        installed = importlib.metadata.version("b12x")
    except importlib.metadata.PackageNotFoundError:
        return False

    try:
        supported_version = Version(installed) == Version(B12X_VERSION)
    except InvalidVersion as exc:
        raise RuntimeError(f"Unsupported b12x version {installed!r}") from exc
    if not supported_version:
        raise RuntimeError(
            f"Unsupported b12x version {installed}; this backend requires "
            f"b12x=={B12X_VERSION}"
        )

    try:
        fused_moe = importlib.import_module("b12x.moe.fused_moe")
    except ImportError as exc:
        raise RuntimeError(
            f"b12x {installed} is installed but does not provide "
            f"'b12x.moe.fused_moe'. This backend requires b12x "
            f"=={B12X_VERSION}, which moved the planned MoE API out of "
            f"the removed 'b12x.integration.tp_moe' module. "
            f"Install b12x=={B12X_VERSION}."
        ) from exc

    expected = {
        "plan_weights": {
            "quant_modes",
            "num_experts",
            "hidden_size",
            "intermediate_size",
            # NVFP4 needs the source layout and the ModelOpt global scales.
            "source_format",
            "w13_layout",
        },
        "prepare_weights": {
            "plan",
            "w1_global_scale",
            "a1_gscale",
            "w2_global_scale",
            "a2_gscale",
        },
        "ExpertWeights": {
            "plan",
            "w1_fp4",
            "w1_blockscale",
            "w2_fp4",
            "w2_blockscale",
        },
        "Caps": {"weight_plan", "max_tokens", "quant_mode"},
        "Plan.bind": {"experts", "scratch"},
    }
    missing = []
    for name, required in expected.items():
        obj: Any = fused_moe
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            missing.append(name)
            continue
        params = set(inspect.signature(obj).parameters)
        if not required <= params:
            missing.append(f"{name}({sorted(required - params)})")
    if missing:
        raise RuntimeError(
            f"b12x {installed} exposes 'b12x.moe.fused_moe' but is missing "
            f"required API: {', '.join(missing)}. Install "
            f"b12x=={B12X_VERSION}."
        )
    return True


class B12xExperts(mk.FusedMoEExpertsModular):
    """Native FP4 MoE backend powered by b12x kernels.

    Handles both native MXFP4 (``w4a16`` over ``fp4_e8m0_k32`` sources) and
    native NVFP4 (``nvfp4`` over ``modelopt_nvfp4`` sources).
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype in _B12X_SOURCE_FORMATS, (
            "B12xExperts only supports native MXFP4/NVFP4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self._experts: Any | None = None
        self._weight_plan: Any | None = None
        self._scratch_plan: Any | None = None
        self._planned_quant_mode: str | None = None
        self._released_w4a16_source_scales = False
        self._unit_scale_by_device: dict[torch.device, torch.Tensor] = {}

    def _weight_dtype(self) -> str:
        return cast(str, self.quant_config.weight_quant_dtype)

    def _source_format(self) -> str:
        return _B12X_SOURCE_FORMATS[self._weight_dtype()]

    def _quant_mode(self) -> str:
        return _B12X_QUANT_MODES[self._weight_dtype()]

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

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        *,
        weight_name: str,
    ) -> torch.Tensor:
        """Return the per-expert weight global scale for the active source.

        MXFP4 sources carry their scale entirely in the E8M0 block scales, so
        b12x expects a unit global scale. ModelOpt NVFP4 sources carry a real
        per-expert global scale that must be forwarded verbatim.

        Args:
            device: Device the prepared weights live on.
            num_experts: Local expert count.
            weight_name: Either ``"w1"`` or ``"w2"``.

        Returns:
            A contiguous float32 tensor of ``num_experts`` scales.

        Raises:
            RuntimeError: When an NVFP4 source is missing its global scales.
            ValueError: When the scale count does not match ``num_experts``.
        """
        if self._source_format() != "modelopt_nvfp4":
            return self._unit_expert_scale(device, num_experts)

        if weight_name == "w1":
            scale = self.g1_alphas
        elif weight_name == "w2":
            scale = self.g2_alphas
        else:
            raise ValueError(f"unknown b12x weight name: {weight_name}")

        if scale is None:
            raise RuntimeError(
                f"B12X ModelOpt NVFP4 MoE requires {weight_name} global scales"
            )
        if int(scale.numel()) != num_experts:
            raise ValueError(
                f"B12X ModelOpt NVFP4 MoE expected {num_experts} "
                f"{weight_name} global scales, got {int(scale.numel())}"
            )
        if scale.device != device:
            _raise_if_capture_copy_required(
                scale, f"{weight_name} global scale device normalization"
            )
            scale = scale.to(device=device)
        if scale.dtype != torch.float32:
            _raise_if_capture_copy_required(
                scale, f"{weight_name} global scale dtype normalization"
            )
            scale = scale.to(torch.float32)
        if not scale.is_contiguous():
            _raise_if_capture_copy_required(
                scale, f"{weight_name} global scale contiguity normalization"
            )
            scale = scale.contiguous()
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
        if not self._plan_discards_source_parameters():
            return

        self._release_w4a16_source_scales(layer)
        self._release_w4a16_source_weights(layer)
        _maybe_release_cuda_cache(device)

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(120) and has_b12x_moe()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in {
            (kMxfp4Static, None),
            (kNvfp4Static, None),
            (kNvfp4Static, kNvfp4Dynamic),
        }

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

        num_experts = int(w1.shape[0])
        unit_scale = self._unit_expert_scale(w1.device, num_experts)
        w1_global_scale = self._weight_global_scale(
            w1.device, num_experts, weight_name="w1"
        )
        w2_global_scale = self._weight_global_scale(
            w2.device, num_experts, weight_name="w2"
        )
        if self._source_format() == "modelopt_nvfp4":
            if self.a1_gscale is None or self.a2_gscale is None:
                raise RuntimeError(
                    "B12X NVFP4 MoE requires a1/a2 activation global scales"
                )
            a1_gscale = _normalize_modelopt_expert_scale(self.a1_gscale)
            a2_gscale = _normalize_modelopt_expert_scale(self.a2_gscale)
        else:
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        from b12x.moe.fused_moe import plan_weights  # type: ignore[import-not-found]

        self._weight_plan = plan_weights(
            quant_modes=self._quant_mode(),
            source_format=self._source_format(),
            w13_layout=self._w13_layout(),
            activation=_b12x_activation_name(activation),
            params_dtype=params_dtype,
            num_experts=num_experts,
            hidden_size=int(self.moe_config.hidden_dim),
            intermediate_size=int(self.moe_config.intermediate_size_per_partition),
        )
        from b12x.moe.fused_moe import prepare_weights  # type: ignore[import-not-found]

        self._experts = prepare_weights(
            plan=self._weight_plan,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_global_scale=w1_global_scale,
            a1_gscale=a1_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_global_scale=w2_global_scale,
            a2_gscale=a2_gscale,
            params_dtype=params_dtype,
        )
        self._plan_scratch(w1.device)
        return self._experts

    def _plan_discards_source_parameters(self) -> bool:
        """Report whether b12x took ownership of the source allocations.

        The b12x planner keeps source storage for NVFP4 recipes and transfers
        it for native W4A16, so the source parameters may only be dropped when
        the plan says the owner supersedes them.

        Returns:
            True when the source parameters are safe to release.

        Raises:
            RuntimeError: When the plan does not expose the ownership contract.
        """
        plan = self._weight_plan
        discards = getattr(plan, "discards_source_parameters", None)
        if not isinstance(discards, bool):
            raise RuntimeError(
                "b12x weight plan does not expose "
                "'discards_source_parameters'; refusing to guess source "
                f"ownership. Install b12x=={B12X_VERSION}."
            )
        return discards

    def _lookup_prepared_w4a16(self) -> Any | None:
        return self._experts

    def _plan_scratch(self, device: torch.device) -> None:
        from b12x.moe.fused_moe import Caps, plan  # type: ignore[import-not-found]

        max_tokens = max(
            1,
            int(self.moe_config.max_num_tokens) * int(self.moe_config.dp_size),
            int(self.moe_config.max_capture_size),
        )
        self._planned_quant_mode = self._quant_mode()
        self._scratch_plan = plan(
            Caps(
                max_tokens=max_tokens,
                num_topk=int(self.moe_config.experts_per_token),
                device=device,
                weight_plan=self._weight_plan,
                quant_mode=self._planned_quant_mode,
                core_token_counts=(max_tokens,),
                route_num_experts=0,
                apply_router_weight_on_input=False,
                swiglu_limit=getattr(self.quant_config, "gemm1_clamp_limit", None),
                frozen=True,
            )
        )

    def _assert_owner_aliases_source(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
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
        if self._scratch_plan is None:
            raise RuntimeError(
                "B12X scratch plan was not prepared before workspace sizing"
            )
        specs = self._scratch_plan.scratch_specs()
        if len(specs) != 1 or specs[0].dtype != torch.uint8:
            raise RuntimeError("B12X scratch plan must expose one uint8 arena")
        scratch_bytes = int(specs[0].shape[0])
        workspace_elements = (scratch_bytes + torch.bfloat16.itemsize - 1) // (
            torch.bfloat16.itemsize
        )
        return (0,), (workspace_elements,), (M, K)

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
        if self._experts is None or self._scratch_plan is None:
            raise RuntimeError(
                "B12X MoE weights and scratch plan were not prepared before execution"
            )

        if expert_map is not None:
            raise RuntimeError(
                "B12X MoE does not support expert_map with the current b12x_moe_fp4 API"
            )
        if apply_router_weight_on_input:
            raise RuntimeError(
                "B12X MoE scratch was planned without input router weighting"
            )
        if workspace2 is None:
            raise RuntimeError("B12X MoE requires its planned modular workspace")

        topk_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        topk_weights = _normalize_b12x_moe_topk_weights(topk_weights)
        scratch = workspace2.view(torch.uint8)
        required_bytes = int(self._scratch_plan.scratch_specs()[0].shape[0])
        scratch = scratch[:required_bytes]
        _run_moe(
            a=hidden_states,
            experts=self._experts,
            scratch_plan=self._scratch_plan,
            scratch=(scratch,),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
            unit_scale_contract=self._planned_quant_mode != "nvfp4",
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
