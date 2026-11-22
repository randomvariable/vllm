# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 output projection with SM100/SM103 and SM12x native paths."""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.pytorch import TorchMxFp8LinearKernel
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer,
    get_b12x_preparation,
    get_b12x_scratch_buffers,
    reuse_packed_weight_storage,
)
from vllm.utils.torch_utils import current_stream, direct_register_custom_op
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource


class B12xWoProjection:
    def __init__(self, api: ModuleType, name: str) -> None:
        self.api = api
        self.name = name
        self.weights: Any = None
        self.plans: dict[int, Any] = {}

    def pack(self, layer: nn.Module) -> None:
        a, b = layer.wo_a, layer.wo_b
        groups = layer.n_local_groups
        # ModelOpt repeats each checkpoint scale across 32 output rows.
        # Take one byte per block; b12x expands/swizzles without requantizing.
        packed = self.api.pack_weights(
            a.weight.detach(),
            a.weight_scale.detach()[::32],
            b.weight.detach(),
            b.weight_scale.detach()[::32],
            groups=groups,
            group_width=layer.n_local_heads // groups * layer.head_dim,
            rank=layer.o_lora_rank,
            hidden=layer.hidden_size,
            block_size=(32, 32),
        )
        self.weights = reuse_packed_weight_storage(self.weights, packed)

    def _rope_args(self, layer: nn.Module) -> dict[str, int]:
        return dict(
            heads_per_group=layer.n_local_heads // layer.n_local_groups,
            nope_dim=layer.nope_head_dim,
            rope_dim=layer.rope_head_dim,
        )

    def get_b12x_preparation_units(
        self, layer: nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        weights = self.weights
        assert weights is not None
        table = layer.rotary_emb.cos_sin_cache
        preparation = get_b12x_preparation()
        assert preparation is not None
        rope = self._rope_args(layer)

        def prepare(state):
            rows = state.query.max_tokens
            source = torch.empty(
                (rows, layer.n_local_heads, layer.head_dim),
                dtype=torch.bfloat16,
                device=table.device,
            )
            positions = torch.zeros(rows, dtype=torch.int64, device=table.device)
            binding = state.bind_inv_rope(
                scratch=tuple(
                    torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                    for spec in state._scratch_state.scratch_specs()
                ),
                o=source,
                positions=positions,
                cos_sin_cache=table,
                weights=weights,
                **rope,
            )
            return preparation.PreparedCall(
                run=lambda: state.run_inv_rope(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(binding,),
            )

        requests = []
        # The pinned API fuses WO-B's input quantization at fixed M <= 8.
        # One dynamic plan handles every remaining decode or prefill size.
        for key in (*range(1, min(8, workload.max_tokens) + 1), 0):
            if key not in self.plans:
                self.plans[key] = self.api.plan(
                    self.api.Caps(
                        device=table.device,
                        max_tokens=key or workload.max_tokens,
                        groups=weights.groups,
                        group_width=weights.group_width,
                        rank=weights.rank,
                        hidden=weights.hidden,
                    ),
                    invocation=dict(
                        operation="inv_rope",
                        dynamic_tokens=key == 0,
                        positions_dtype="int64",
                        cos_sin_dtype=str(table.dtype).removeprefix("torch."),
                        **rope,
                    ),
                )
            requests.append(
                self.plans[key].request(
                    name=f"{self.name}.wo.{key or 'dynamic'}",
                    prepare_call=prepare,
                    benchmark_call=prepare,
                )
            )
        return (
            B12xPreparationUnit(
                name="DSV41 WO projection",
                key=self.name,
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def run(self, layer: nn.Module, o: torch.Tensor, positions: torch.Tensor):
        if o.shape[0] == 0:
            return o.new_empty((0, layer.hidden_size))
        plan = self.plans.get(o.shape[0], self.plans[0])
        assert plan.prepared is not None, "DSV41 WO plan must be prepared before use"
        scratch = get_b12x_scratch_buffers(plan)
        binding = self.api.bind_inv_rope(
            plan,
            scratch=scratch,
            o=o[:, : layer.n_local_heads],
            positions=positions,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            weights=self.weights,
            **self._rope_args(layer),
        )
        retain_cuda_graph_capture_resource(scratch)
        return self.api.run_inv_rope(
            binding=binding, plan=plan, stream=current_stream().cuda_stream
        )


def _run_b12x_wo_projection(
    o: torch.Tensor, positions: torch.Tensor, hidden: int, layer_name: str
) -> torch.Tensor:
    layer = b12x_layer(layer_name)
    return layer._b12x_wo.run(layer, o, positions)


def _fake_b12x_wo_projection(
    o: torch.Tensor, positions: torch.Tensor, hidden: int, layer_name: str
) -> torch.Tensor:
    return o.new_empty((o.shape[0], hidden))


direct_register_custom_op(
    op_name="dsv41_b12x_wo_projection",
    op_func=_run_b12x_wo_projection,
    fake_impl=_fake_b12x_wo_projection,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _fused_wo_a_max_tokens(layer: nn.Module) -> int:
    max_tokens = getattr(layer, "_fused_wo_a_token_limit", None)
    if max_tokens is None:
        # Additional token tiles only pay off for smaller grids.
        ctas_per_tile = layer.n_local_heads * (layer.o_lora_rank // 128)
        max_tokens = 96 if ctas_per_tile <= 128 else 32
        layer._fused_wo_a_token_limit = max_tokens
    return max_tokens


def _can_fuse_wo_a(layer: nn.Module) -> bool:
    """Whether the attention layer matches the fused WO-A kernel's layout."""
    # The fused kernel never materializes the BF16 input an adapter on wo_a
    # needs, so such a layer takes ``deep_gemm_fp8_o_proj`` instead.
    if getattr(layer.wo_a, "requires_unquantized_input", False):
        return False
    method = getattr(layer.wo_b, "scheme", layer.wo_b.quant_method)
    return (
        current_platform.is_device_capability_family(100)
        # WO-A is FP8 with per-32 scales, and WO-B takes MXFP8 input directly.
        and layer._einsum_recipe == (1, 1, 32)
        and get_input_quant_key(layer.wo_b) == kMxfp8Dynamic
        and isinstance(
            getattr(method, "kernel", None),
            (FlashInferCutedslMxfp8LinearKernel, TorchMxFp8LinearKernel),
        )
        # A group's heads form one portable (<= 8 CTA) cluster.
        and layer.n_local_heads // layer.n_local_groups <= 8
        and (layer.nope_head_dim, layer.rope_head_dim) == (448, 64)
        and layer.o_lora_rank % 128 == 0
    )


def _fused_wo_a_shape(layer: nn.Module) -> dict[str, int]:
    """The fused WO-A kernel's warmup arguments for `layer`."""
    return dict(
        max_tokens=_fused_wo_a_max_tokens(layer),
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        o_lora_rank=layer.o_lora_rank,
    )


def precompile_dsv41_o_proj_early_trigger(layer: nn.Module, max_tokens: int) -> None:
    """Compile the fused WO-A variant `dsv41_o_proj` dispatches with
    `early_trigger`, for its token counts up to `max_tokens`."""
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        shape = _fused_wo_a_shape(layer)
        shape["max_tokens"] = min(shape["max_tokens"], max_tokens)
        _FUSED_WO_A_KERNEL.warmup(**shape, early_trigger=True)


def register_dsv41_o_proj_warmup(layer: nn.Module) -> None:
    """Warm the fused WO-A token counts `dsv41_o_proj` can dispatch."""
    # Draft models load outside the warmup registry; their layers share the
    # target's attention shapes, so the target's registration covers them too.
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        _FUSED_WO_A_KERNEL.register_warmup(**_fused_wo_a_shape(layer))


def dsv41_o_proj(
    layer: nn.Module,
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    wo_b_proj: Callable[..., torch.Tensor] | None = None,
    early_trigger: bool = False,
) -> torch.Tensor:
    """`deep_gemm_fp8_o_proj` with WO-A fused for small SM100/SM103 batches;
    `wo_b_proj` replaces `layer._wo_b_proj`. With `early_trigger`, the
    fused WO-A launches `wo_b_proj` (PDL) right after its TMEM allocation,
    for a `wo_b_proj` that loads its weights before its grid wait."""
    wo_b_proj = wo_b_proj or layer._wo_b_proj
    o = attn_out[:, : layer.n_local_heads, :]
    cos_sin_cache = layer.rotary_emb.cos_sin_cache
    if 1 <= o.shape[0] <= _fused_wo_a_max_tokens(layer) and _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        q, scales = _FUSED_WO_A_KERNEL(
            x=o,
            positions=positions,
            rope=cos_sin_cache,
            weight=layer.wo_a.weight,
            weight_scale=layer.wo_a.weight_scale,
            early_trigger=early_trigger,
        )
        return wo_b_proj(
            QuantizedActivation(q, scales, o.dtype, q.shape, kMxfp8Dynamic)
        )
    z = layer.wo_a.grouped_forward(
        o,
        positions,
        cos_sin_cache,
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        nope_dim=layer.nope_head_dim,
        rope_dim=layer.rope_head_dim,
        o_lora_rank=layer.o_lora_rank,
        einsum_recipe=layer._einsum_recipe,
        tma_aligned_scales=layer._tma_aligned_scales,
    )
    return wo_b_proj(z.flatten(1))
