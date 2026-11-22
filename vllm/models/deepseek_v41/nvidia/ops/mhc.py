# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""

from collections.abc import Callable
from functools import cache, partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_delayed_tilelang,
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    get_b12x_mhc,
    get_b12x_preparation,
    get_b12x_scratch_buffers,
    set_b12x_preparation_provider,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource

from .mega_mhc import (
    can_use_mega_mhc,
    mhc_shifted_post_pre_deep_gemm,
)

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    from ..specialized_decode.ampere.all_reduce import AmpereAllReduce
    from .cute_dsl import AllReduceMHC

logger = init_logger(__name__)

# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.
MHC_OVERLAP_MAX_TOKENS = 16
# Some b12x mHC kernels put the token dimension on CUDA grid Y or Z.
B12X_MHC_MAX_TOKENS = 65_535

# The fused all-reduce + mHC kernel of the overlap path, built once by
# init_mhc_all_reduce.
_all_reduce_mhc: "AllReduceMHC | None" = None
# The SM80 boundaries' two-shot Lamport all-reduce, built once by
# init_ampere_all_reduce.
_ampere_all_reduce: "AmpereAllReduce | None" = None


def _b12x_mhc_numerics(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    post_mult: float,
    iters: int,
) -> bool:
    # The pinned native finalizer specializes these values and uses one HC eps.
    return (
        rms_eps in (1e-20, 1e-6, 1e-5)
        and hc_pre_eps == hc_sinkhorn_eps == 1e-6
        and post_mult == 2.0
        and iters == 20
    )


class B12xMHC:
    """Prepared lagged mHC plans owned by a target or draft decoder layer."""

    def __init__(self, api: ModuleType) -> None:
        self.api = api
        self._plans: dict[str, dict[int, Any]] = {}

    @staticmethod
    def for_layer(layer) -> "B12xMHC | None":
        if not (
            current_platform.is_device_capability_family(120)
            and layer.hc_mult == 4
            and layer.hidden_size in (4096, 5120, 7168)
        ):
            return None
        api = get_b12x_mhc()
        if api is None or get_b12x_preparation() is None:
            return None
        try:
            if not api.is_supported():
                return None
        except ImportError:
            return None
        provider = B12xMHC(api)
        set_b12x_preparation_provider(layer, provider)
        return provider

    @staticmethod
    def supports_lagged(layer, norm) -> bool:
        return norm.weight.dtype in (torch.bfloat16, torch.float32) and (
            _b12x_mhc_numerics(
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
            )
        )

    def get_b12x_preparation_units(
        self, layer, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        device = layer.hc_attn_fn.device
        if device.type != "cuda":
            return ()
        declarations: dict[str, dict[str, Any]] = {
            "post": dict(operation="post", output_mode="functional"),
            "collapse": dict(operation="collapse", collapse_weighted=True),
            "mean": dict(operation="collapse"),
        }
        for name, norm in (("attn", layer.attn_norm), ("ffn", layer.ffn_norm)):
            if not self.supports_lagged(layer, norm):
                continue
            common = dict(
                output_mode="provided",
                lagged_mix=True,
                has_norm_weight=True,
                norm_weight_dtype=str(norm.weight.dtype).removeprefix("torch."),
                rms_eps=layer.rms_norm_eps,
                hc_eps=layer.hc_eps,
                sinkhorn_iters=layer.hc_sinkhorn_iters,
                norm_eps=norm.variance_epsilon,
            )
            declarations[name] = dict(**common, operation="post_pre")
            if name == "attn":
                declarations["pre"] = dict(
                    **common, operation="pre", expanded_residual=True
                )
                if layer.hc_attn_fn_broadcast is not None:
                    declarations["broadcast"] = dict(**common, operation="pre")

        max_tokens = min(workload.max_tokens, B12X_MHC_MAX_TOKENS)
        requests = []
        for name, invocation in declarations.items():
            # The lagged projection uses small-M tactics. Post and collapse
            # have dynamic M and only need the max-capacity plan.
            lagged = invocation.get("lagged_mix", False)
            capacities = (max_tokens,)
            if lagged:
                capacities = (*range(1, min(9, max_tokens)), *capacities)
            plans = self._plans.setdefault(name, {})
            for tokens in capacities:
                if tokens not in plans:
                    plans[tokens] = self.api.plan(
                        self.api.Caps(
                            device=device,
                            max_tokens=tokens,
                            hidden_size=layer.hidden_size,
                        ),
                        invocation=invocation,
                    )
                prepare = self._prepare_call(layer, name, tokens if lagged else 1)
                requests.append(
                    plans[tokens].request(
                        name=f"dsv41.mhc.{id(layer):x}.{name}.m{tokens}",
                        prepare_call=prepare,
                        benchmark_call=prepare,
                    )
                )
        return (
            B12xPreparationUnit(
                name="V41MHC",
                key=(id(layer), workload.max_tokens, workload.fixed_token_counts),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def _prepare_call(self, layer, name, tokens):
        def prepare(state):
            # As in the fork, prime the selected implementation directly: its
            # public entry point requires the plan to have been published.
            impl = self.api._impl
            query = state.query
            fn = layer.hc_ffn_fn if name == "ffn" else layer.hc_attn_fn
            device = fn.device
            h = layer.hidden_size
            x = torch.zeros((tokens, h), dtype=torch.bfloat16, device=device)
            residual = torch.zeros((tokens, 4, h), dtype=x.dtype, device=device)
            mix = torch.zeros((tokens, 4), dtype=torch.float32, device=device)
            comb = None
            prev_post = None

            def produce():
                residual.normal_(std=0.25)
                if query.operation != "collapse":
                    x.normal_(std=0.25)
                mix.zero_()
                mix[:, 0].fill_(1)
                if comb is not None:
                    comb.normal_(mean=0.25, std=0.05)
                if prev_post is not None:
                    prev_post.normal_(mean=1.0, std=0.1)

            if query.operation == "collapse":
                run = partial(
                    impl._run_collapse_impl,
                    residual,
                    mix if query.collapse_weighted else None,
                    out=x,
                    _state=state,
                )
            else:
                comb = torch.zeros((tokens, 4, 4), dtype=torch.float32, device=device)
                if query.operation == "post":
                    run = partial(
                        impl._b12x_mhc_post_impl, x, residual, mix, comb, _state=state
                    )
                else:
                    is_ffn = name == "ffn"
                    if name == "broadcast":
                        fn = layer.hc_attn_fn_broadcast
                    args = (
                        fn,
                        layer.hc_ffn_scale if is_ffn else layer.hc_attn_scale,
                        layer.hc_ffn_base if is_ffn else layer.hc_attn_base,
                    )
                    scratch = [
                        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                        for spec in state.scratch_specs()
                    ]
                    out, post, comb_out, y, pre_out = _b12x_mhc_outputs(residual)
                    binding = state.bind(
                        scratch=scratch,
                        tokens=tokens,
                        out=out,
                        post=post,
                        comb=comb_out,
                        y=y,
                        pre_out=pre_out,
                    )
                    kwargs = dict(
                        rms_eps=query.rms_eps,
                        hc_eps=query.hc_eps,
                        sinkhorn_iters=query.sinkhorn_iters,
                        norm_eps=query.norm_eps,
                        norm_weight=(
                            layer.ffn_norm if is_ffn else layer.attn_norm
                        ).weight,
                        pre_mix=mix,
                        binding=binding,
                        _state=state,
                    )
                    if query.operation == "pre":
                        run = partial(
                            impl._b12x_mhc_pre_impl,
                            residual if query.expanded_residual else x,
                            *args,
                            **kwargs,
                        )
                    else:
                        prev_post = torch.empty_like(mix)
                        run = partial(
                            impl._b12x_mhc_post_pre_impl,
                            x,
                            residual,
                            prev_post,
                            comb,
                            *args,
                            **kwargs,
                        )
            preparation = get_b12x_preparation()
            assert preparation is not None
            return preparation.PreparedCall(run=run, produce=produce)

        return prepare

    def plan_for(self, name: str, residual: torch.Tensor) -> Any | None:
        if (
            not residual.is_cuda
            or residual.dtype != torch.bfloat16
            or not residual.is_contiguous()
            or (residual.ndim == 3 and residual.shape[1] != 4)
        ):
            return None
        plans = self._plans.get(name)
        if not plans:
            return None
        tokens = residual.shape[0]
        if tokens in plans and plans[tokens].prepared is not None:
            return plans[tokens]
        capacity = max(plans)
        plan = plans[capacity]
        if tokens <= capacity and plan.prepared is not None:
            return plan
        return None


def mhc_post(x, residual, post_mix, comb, *, b12x: B12xMHC | None):
    plan = b12x.plan_for("post", residual) if b12x is not None else None
    if plan is not None:
        assert b12x is not None
        return b12x.api.run_post(x, residual, post_mix, comb, plan=plan)
    return mhc_post_tilelang(x, residual, post_mix, comb)


def mhc_collapse(residual, pre_mix, *, b12x: B12xMHC | None):
    name = "mean" if pre_mix is None else "collapse"
    plan = b12x.plan_for(name, residual) if b12x is not None else None
    if plan is not None and (pre_mix is None or pre_mix.is_contiguous()):
        assert b12x is not None
        out = residual.new_empty((residual.shape[0], residual.shape[-1]))
        return b12x.api.run_collapse(residual, pre_mix, out=out, plan=plan)
    if pre_mix is None:
        return residual.mean(dim=1)
    return hc_collapse_triton(residual, pre_mix)


def _b12x_mhc_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, hidden = residual.shape[0], residual.shape[-1]
    return (
        residual.new_empty((rows, 4, hidden)),
        residual.new_empty((rows, 4), dtype=torch.float32),
        residual.new_empty((rows, 4, 4), dtype=torch.float32),
        residual.new_empty((rows, hidden)),
        residual.new_empty((rows, 4), dtype=torch.float32),
    )


def _run_b12x_lagged_mhc(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    pre_mix: torch.Tensor | None,
    plan_handle: int,
    x: torch.Tensor | None = None,
    prev_post: torch.Tensor | None = None,
    prev_comb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = _b12x_mhc_outputs(residual)
    rows = residual.shape[0]
    if rows == 0:
        return outputs

    api = get_b12x_mhc()
    preparation = get_b12x_preparation()
    assert api is not None and preparation is not None
    plan = preparation.plan_from_handle(plan_handle)
    if pre_mix is None:
        pre_mix = residual.new_zeros((rows, 4), dtype=torch.float32)
        pre_mix[:, 0].fill_(1)

    out, post, comb, y, pre_out = outputs
    scratch = get_b12x_scratch_buffers(plan)
    binding = api.bind(
        plan,
        scratch=scratch,
        tokens=rows,
        out=out,
        post=post,
        comb=comb,
        y=y,
        pre_out=pre_out,
    )
    retain_cuda_graph_capture_resource(scratch)
    query = plan.query
    kwargs = dict(
        binding=binding,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=query.norm_eps,
        rms_eps=query.rms_eps,
        hc_eps=query.hc_eps,
        sinkhorn_iters=query.sinkhorn_iters,
    )
    # norm.mhc includes the FP32 projection and RMS stats. The BF16
    # gemm.hyperconnection_gate computes a different gate-and-mean operation.
    if x is None:
        api.run_pre(residual, fn, hc_scale, hc_base, **kwargs)
    else:
        assert prev_post is not None and prev_comb is not None
        api.run_post_pre(
            x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, **kwargs
        )
    return outputs


def _fake_b12x_lagged_mhc(
    residual,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    pre_mix,
    plan_handle,
    x=None,
    prev_post=None,
    prev_comb=None,
):
    return _b12x_mhc_outputs(residual)


direct_register_custom_op(
    op_name="dsv41_b12x_lagged_mhc",
    op_func=_run_b12x_lagged_mhc,
    fake_impl=_fake_b12x_lagged_mhc,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _b12x_pre_plan(
    b12x: B12xMHC | None,
    name: str,
    residual: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> Any | None:
    if (
        b12x is None
        or norm_weight is None
        or not _b12x_mhc_numerics(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, post_mult, sinkhorn_repeat
        )
    ):
        return None
    plan = b12x.plan_for(name, residual)
    if plan is not None and (
        plan.query.rms_eps == rms_eps
        and plan.query.norm_eps == norm_eps
        and plan.query.norm_weight_dtype
        == str(norm_weight.dtype).removeprefix("torch.")
    ):
        return plan
    return None


def mhc_pre_delayed(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    hc_mult: int,
    stream: torch.cuda.Stream | None,
    b12x: B12xMHC | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run lagged pre, expanding a broadcast embedding as part of the boundary."""
    broadcast = residual.ndim == 2
    plan = _b12x_pre_plan(
        b12x if hc_mult == 4 else None,
        "broadcast" if broadcast else "pre",
        residual,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )
    if plan is not None:
        assert norm_weight is not None
        residual_out, post, comb, y, next_pre = torch.ops.vllm.dsv41_b12x_lagged_mhc(
            residual, fn, hc_scale, hc_base, norm_weight, pre_mix, plan.handle
        )
        return residual_out, post.unsqueeze(-1), comb, y, next_pre

    x = residual if broadcast else None
    if broadcast:
        residual = hc_expand(residual, hc_mult)
    pre: Callable[..., tuple[Any, Any, Any, Any]]
    if use_ampere_mhc():
        pre = partial(mhc_pre_delayed_ampere, stream=stream)
    elif stream is not None:
        pre = partial(mhc_pre_delayed_overlap, stream=stream)
    else:
        pre = mhc_pre_delayed_tilelang
    return residual, *pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        x=x,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Check kernel requirements and safety of sharing the coefficient stream."""
    config = vllm_config.model_config.hf_config
    # DeepGEMM's prenorm kernel requires K % 64 == 0, N % 8 == 0, and N <= 32.
    mix_size = config.hc_mult * (config.hc_mult + 2)
    return (
        current_platform.is_device_capability_family(100)
        and is_deep_gemm_supported()
        and config.hidden_size % 64 == 0
        and 0 < mix_size <= 32
        and mix_size % 8 == 0
        and not vllm_config.parallel_config.use_ubatching
    )


# The SM80 mHC kernels (specialized_decode/ampere/mhc.py) take boundaries of
# up to this many rows; larger ones keep the TileLang kernels.
AMPERE_MHC_MAX_TOKENS = 256
# SM90 runs the same kernels up to this many rows, with the coefficients on
# CUDA cores (see ``_uses_mma``): on H100 at TP8 with five draft tokens they
# beat the stock boundary, whose projection runs on DeepGEMM's tensor-core
# kernel, at 48 rows and lost from 96.
SM90_MHC_MAX_TOKENS = 48
# The fused all-reduce is one-shot, as FlashInfer's MNNVL one-shot kernel,
# which vLLM's all-reduce leaves for its two-shot one above 1 MiB of all
# ranks' rows (MNNVL_ONE_SHOT_THRESHOLD): 12 rows of DSV4.1 at TP8. On H100 at
# TP8 with five draft tokens the fused boundary takes 2.5% off a decode step
# at 6 rows, and is 1.2% and 3.0% slower than the two-shot one at 24 and 48.
SM90_MHC_ALL_REDUCE_MAX_TOKENS = 12


@cache
def ampere_mhc_max_tokens() -> int:
    """The most rows an mHC boundary takes the SM80 CuTe DSL kernels for:
    all of them on SM80, small decode batches on SM90, none elsewhere."""
    # Imported here: the specialized_decode package imports this module.
    from ..specialized_decode.ampere import is_ampere

    if is_ampere():
        return AMPERE_MHC_MAX_TOKENS
    if current_platform.is_cuda() and current_platform.is_device_capability(90):
        return SM90_MHC_MAX_TOKENS
    return 0


def use_ampere_mhc() -> bool:
    """Whether any mHC boundary takes the SM80 CuTe DSL kernels."""
    return ampere_mhc_max_tokens() > 0


def supports_ampere_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Whether FULL graphs on SM80 and SM90 compute the mHC coefficients on
    a side stream: only for a hidden size the SM80 kernels take (otherwise
    the TileLang kernels run in order), and not with micro-batching, which
    the single ticket workspace rules out."""
    if not use_ampere_mhc() or vllm_config.parallel_config.use_ubatching:
        return False
    from ..specialized_decode.ampere.mhc import HC, supports

    return supports(1, vllm_config.model_config.hf_text_config.hidden_size, HC)


def prepare_ampere_mhc(hidden_size: int) -> None:
    """Allocate the SM80 mHC kernels' persistent buffers on the current
    device and compile them, before any capture."""
    from ..specialized_decode.ampere.mhc import HC, MAX_ROWS, prepare, supports

    assert MAX_ROWS == AMPERE_MHC_MAX_TOKENS
    if supports(1, hidden_size, HC):
        prepare(torch.accelerator.current_device_index(), hidden_size)


def _ampere_mhc(residual: torch.Tensor, norm_weight: torch.Tensor | None) -> bool:
    """Whether a boundary over `residual` takes the SM80 CuTe DSL kernels."""
    if norm_weight is None or not 0 < residual.shape[0] <= ampere_mhc_max_tokens():
        return False
    from ..specialized_decode.ampere.mhc import supports

    return supports(residual.shape[0], residual.shape[-1], residual.shape[1])


def ampere_projection_splits() -> tuple[int, ...]:
    """The split counts of the SM80 prefill projection (`_mhc_pre_split`),
    for the stock epilogue's warmup: none off SM80."""
    from ..specialized_decode.ampere import is_ampere

    if not is_ampere():
        return ()
    from ..specialized_decode.ampere.mhc_projection import SPLITS

    return (SPLITS,)


def _split_projection(
    residual: torch.Tensor, width: int, norm_weight: torch.Tensor | None
) -> bool:
    """Whether a boundary over `residual` projects its `width`-wide rows
    on the SM80 split-k kernel: on SM80, above the rows the SM80 boundary
    kernels take."""
    if norm_weight is None or residual.shape[0] <= AMPERE_MHC_MAX_TOKENS:
        return False
    if not ampere_projection_splits():
        return False
    from ..specialized_decode.ampere.mhc_projection import supports

    return supports(width)


def _mhc_pre_split(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    x: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    capture_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mhc_pre_delayed_tilelang` with a norm weight on SM80 prefill: the
    projection on the split-k tensor-core kernel (mhc_projection.py), then
    the stock epilogue. Returns the post-, comb-mix, layer input, next
    pre-mix and the stream mean (empty unless `capture_aux`)."""
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

    from ..specialized_decode.ampere.mhc_projection import mhc_projection

    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    mixes, sqrsum = mhc_projection(x, fn)
    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    aux = torch.empty(
        num_tokens if capture_aux else 0,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    MHC_PRE_NORM_KERNEL(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        aux if capture_aux else layer_input,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
        write_aux=capture_aux,
    )
    return (
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
        aux,
    )


def mhc_pre_delayed_ampere(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mhc_pre_delayed_tilelang` on SM80: the CuTe DSL kernels, with the
    coefficients on `stream` when given (join it before reading them)."""
    if not _ampere_mhc(residual, norm_weight):
        width = residual.shape[1] * residual.shape[2] if x is None else x.shape[1]
        if _split_projection(residual, width, norm_weight):
            assert norm_weight is not None
            return _mhc_pre_split(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix,
                x,
                norm_weight,
                norm_eps,
            )[:4]
        return mhc_pre_delayed_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            x=x,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    from ..specialized_decode.ampere.mhc import mhc_pre

    assert norm_weight is not None
    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix,
        x,
        norm_weight,
        norm_eps,
        stream=stream,
    )


def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:
    """Whether the mHC boundaries fuse the TP all-reduce (`AllReduceMHC`):
    at TP4 on SM100, whose overlap path it serves, and at TP8 on SM90, where
    the boundaries of the SM80 kernels' path take it; both need NVLink
    multicast."""
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if current_platform.is_device_capability(90):
        tp_size = 8
    elif current_platform.is_device_capability_family(100):
        tp_size = 4
    else:
        return False
    if (
        parallel.tensor_parallel_size != tp_size
        or parallel.enable_expert_parallel
        or config.hidden_size != 5120
        or config.hc_mult != 4
    ):
        return False
    # The custom all-reduce's MNNVL buffers succeed exactly when NVLink
    # multicast does, which the fused kernel's own mailbox needs too.
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)


def mhc_all_reduce_max_tokens() -> int:
    """The most rows the fused all-reduce takes, and so the most a MoE may
    leave its top-k reduction to it for: `SM90_MHC_ALL_REDUCE_MAX_TOKENS`
    on SM90 (its boundaries up to `SM90_MHC_MAX_TOKENS` rows all-reduce
    first), the two-shot kernel's on SM80, `MHC_OVERLAP_MAX_TOKENS` on
    SM100."""
    if _ampere_all_reduce is not None:
        return min(ampere_mhc_max_tokens(), _ampere_all_reduce.capacity)
    if use_ampere_mhc():
        return min(ampere_mhc_max_tokens(), SM90_MHC_ALL_REDUCE_MAX_TOKENS)
    return MHC_OVERLAP_MAX_TOKENS


def init_mhc_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the fused all-reduce + mHC kernel the overlap path reduces with.

    Collective over the TP group, so every rank must call it, and only when
    `supports_mhc_all_reduce` holds.
    """
    global _all_reduce_mhc
    from .cute_dsl import AllReduceMHC

    config = vllm_config.model_config.hf_config
    max_num_tokens = mhc_all_reduce_max_tokens()
    _all_reduce_mhc = AllReduceMHC(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        max_num_tokens=max_num_tokens,
        top_k=config.num_experts_per_tok,
        device=current_platform.current_device(),
    )
    logger.info_once(
        "DSV4.1 mHC: CuTe DSL Lamport all-reduce fused with mHC for up to %d tokens.",
        max_num_tokens,
    )


def supports_ampere_all_reduce(vllm_config: VllmConfig) -> bool:
    """Whether the SM80 mHC boundaries take the TP all-reduce with their own
    kernel (`AmpereAllReduce`, a two-shot all-reduce that stores through
    peer mappings): on SM80, which has no NVLink multicast, at TP 2, 4 or 8
    where vLLM's custom all-reduce runs, so the GPUs are fully connected
    with peer access."""
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    tp_size = parallel.tensor_parallel_size
    if (
        not current_platform.is_device_capability(80)
        or tp_size not in (2, 4, 8)
        or parallel.enable_expert_parallel
        or config.hidden_size % (8 * tp_size)
    ):
        return False
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and not comm.disabled and comm.fully_connected


def init_ampere_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the SM80 boundaries' all-reduce. Collective over the TP group,
    so every rank must call it, and only when `supports_ampere_all_reduce`
    holds."""
    global _ampere_all_reduce
    from ..specialized_decode.ampere.all_reduce import AmpereAllReduce

    group = get_tp_group()
    config = vllm_config.model_config.hf_config
    _ampere_all_reduce = AmpereAllReduce(
        group.device_group,
        config.hidden_size,
        group.device,
        top_k=config.num_experts_per_tok,
    )
    logger.info_once(
        "DSV4.1 mHC: CuTe DSL two-shot Lamport all-reduce for up to %d tokens.",
        _ampere_all_reduce.capacity,
    )


def mhc_all_reduce_ready() -> bool:
    """Whether `init_mhc_all_reduce` has built the fused all-reduce (the
    target model does; its DSpark draft model reuses it)."""
    return _all_reduce_mhc is not None


def ampere_all_reduce_ready() -> bool:
    """Whether `init_ampere_all_reduce` has built the SM80 two-shot kernel
    (the target model does; its DSpark draft model reuses it)."""
    return _ampere_all_reduce is not None


def boundary_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """The TP all-reduce of a partial sum the model reduces itself (a
    sublayer output, the embedding's vocabulary shards): the SM80 two-shot
    Lamport kernel for the rows it takes, else the TP group's."""
    if _ampere_all_reduce is not None and _ampere_all_reduce.supports(x):
        return _ampere_all_reduce(x)
    return get_tp_group().all_reduce(x)


def mhc_pre_delayed_overlap(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    stream: torch.cuda.Stream,
    layer_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the input on the caller stream and coefficients on another stream.

    Returns post mix, residual mix, normalized input, and next pre-mix. Only the
    input is ready on the caller stream; join stream before using coefficients.
    The caller must retain the weights and coefficient outputs until the join.
    """
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    assert residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert norm_weight is not None
    if x is None:
        x = residual.view(n, hc * hidden)
    assert x.is_contiguous()
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    compute_input = layer_input is None
    if layer_input is None:
        layer_input = torch.empty(
            (n, hidden), device=residual.device, dtype=torch.bfloat16
        )
    outputs = post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre
    if n == 0:
        return outputs
    splits = compute_mhc_pre_num_splits(x.shape[1], n)
    mix = torch.empty(
        (splits, n, hc * (hc + 2)), device=residual.device, dtype=torch.float32
    )
    sqr = torch.empty((splits, n), device=residual.device, dtype=torch.float32)
    epilogue = partial(
        MHC_PRE_NORM_KERNEL,
        mix,
        sqr,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre,
        layer_input,  # Unused aux output in split modes.
        hidden_size=hidden,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
    )
    main = torch.cuda.current_stream()
    # Prioritize input readiness for small batches before releasing statistics.
    if compute_input and n <= 8:
        epilogue(split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        epilogue(split_mode="stats")
    for tensor in (x, mix, sqr):
        tensor.record_stream(stream)
    if compute_input and n > 8:
        epilogue(split_mode="input")
    return outputs


def mhc_boundary_weight(
    hidden_size: int, hc_mult: int, device: torch.device
) -> torch.Tensor | None:
    """An empty `prepare_boundary_weight` buffer, or None where the CuTe DSL mHC
    boundary (`mhc_boundary.py`) cannot run."""
    if not (current_platform.is_device_capability_family(100) and has_cutedsl()):
        return None
    from .mhc_boundary import MhcBoundaryKernel, boundary_weight_shape

    if not MhcBoundaryKernel.supports(hidden_size, hc_mult):
        return None
    return torch.empty(
        boundary_weight_shape(hidden_size), dtype=torch.int32, device=device
    )


@cache
def _mhc_shifted_post_pre_boundary():
    from .mhc_boundary import mhc_shifted_post_pre_boundary

    return mhc_shifted_post_pre_boundary


def mhc_shifted_post_pre(
    x: torch.Tensor | MoEOutput,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
    boundary_weight: torch.Tensor | None = None,
    b12x: B12xMHC | None = None,
    b12x_plan_name: str = "attn",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Dispatch shifted post/pre to b12x, overlap, Mega-mHC, or TileLang.

    `boundary_weight` (`fn` laid out by `prepare_boundary_weight`, see
    `mhc_boundary_weight`) runs decode-size Mega-mHC batches on the CuTe DSL kernel.

    A MoE output left unfinalized is finalized inside the fused all-reduce.
    When stream is supplied, join it before consuming the returned coefficients.
    """
    if (
        _ampere_mhc(residual, norm_weight)
        and _all_reduce_mhc is not None
        and (
            isinstance(x, MoEOutput)
            or (
                reduce_results
                and stream is not None
                and residual.shape[0] <= mhc_all_reduce_max_tokens()
            )
        )
    ):
        # SM90 in FULL graphs up to the fused kernel's rows (and wherever a
        # MoE left its top-k reduction open, which it does only for those):
        # the all-reduce, post-mix, collapse and RMSNorm in the fused kernel,
        # then the next coefficients from its streams on the side stream, or
        # inline without one.
        from ..specialized_decode.ampere.mhc import mhc_coefficients

        assert pre_mix is not None and norm_weight is not None
        if isinstance(x, MoEOutput):
            routed = x.routed
            assert isinstance(routed, UnfinalizedMoEOutput)
            assert x.shared_output is not None
            residual, layer_input = _all_reduce_mhc.finalize(
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
                x.shared_output,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        else:
            residual, layer_input = _all_reduce_mhc(
                x.contiguous(),
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        assert layer_input is not None
        post, comb, pre = mhc_coefficients(
            None,
            residual,
            None,
            None,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            stream=stream,
        )
        aux = (
            residual.mean(dim=1)
            if capture_aux
            else residual.new_empty(0, residual.shape[-1])
        )
        return residual, post.unsqueeze(-1), comb, layer_input, pre, aux
    if _ampere_mhc(residual, norm_weight) and (
        isinstance(x, torch.Tensor) or _ampere_all_reduce is not None
    ):
        from ..specialized_decode.ampere.mhc import mhc_coefficients, mhc_post_pre

        assert norm_weight is not None
        if isinstance(x, MoEOutput) or (
            reduce_results
            and _ampere_all_reduce is not None
            and _ampere_all_reduce.supports(x)
        ):
            # The two-shot all-reduce, post-mix, collapse and RMSNorm in one
            # kernel (with the MoE's top-k sum and shared-expert add where
            # the MoE left them open), then the next coefficients from its
            # streams.
            assert _ampere_all_reduce is not None
            if isinstance(x, MoEOutput):
                routed = x.routed
                assert isinstance(routed, UnfinalizedMoEOutput) and routed.route_major
                assert x.shared_output is not None
                residual, layer_input, aux = _ampere_all_reduce.finalize_post_collapse(
                    routed.gemm2_permuted,
                    routed.expert_weights,
                    x.shared_output,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    pre_mix,
                    norm_weight,
                    norm_eps,
                    capture_aux,
                )
            else:
                residual, layer_input, aux = _ampere_all_reduce.post_collapse(
                    x.contiguous(),
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    pre_mix,
                    norm_weight,
                    norm_eps,
                    capture_aux,
                )
            post, comb, pre = mhc_coefficients(
                None,
                residual,
                None,
                None,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                stream=stream,
            )
            return residual, post.unsqueeze(-1), comb, layer_input, pre, aux
        if reduce_results:
            x = boundary_all_reduce(x)
        return mhc_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix,
            norm_weight,
            norm_eps,
            capture_aux,
            stream=stream,
        )
    layer_input = None
    if isinstance(x, MoEOutput):
        assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
        assert pre_mix is not None and norm_weight is not None
        routed = x.routed
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert x.shared_output is not None
        residual, layer_input = _all_reduce_mhc.finalize(
            routed.gemm2_permuted,
            routed.expert_weights,
            routed.expanded_idx_to_permuted_idx,
            x.shared_output,
            residual,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            norm_weight,
            norm_eps,
        )
        if stream is None:
            # Outside FULL graphs, compute the next mixes inline.
            stream = torch.cuda.current_stream()
    elif reduce_results:
        if (
            stream is not None
            and _all_reduce_mhc is not None
            and 0 < x.shape[0] <= MHC_OVERLAP_MAX_TOKENS
        ):
            assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
            assert pre_mix is not None and norm_weight is not None
            residual, layer_input = _all_reduce_mhc(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        else:
            x = get_tp_group().all_reduce(x)
    if stream is not None:
        if layer_input is None:
            assert isinstance(x, torch.Tensor)
            residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        aux = (
            residual.mean(dim=1)
            if capture_aux
            else residual.new_empty(0, residual.shape[-1])
        )
        pre_outputs = mhc_pre_delayed_overlap(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            stream=stream,
            layer_input=layer_input,
        )
        return residual, *pre_outputs, aux

    assert isinstance(x, torch.Tensor)
    plan = _b12x_pre_plan(
        b12x,
        b12x_plan_name,
        residual,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )
    if plan is not None:
        assert norm_weight is not None
        residual, post, comb, y, next_pre = torch.ops.vllm.dsv41_b12x_lagged_mhc(
            residual,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            pre_mix,
            plan.handle,
            x,
            post_layer_mix,
            comb_res_mix,
        )
        aux = (
            mhc_collapse(residual, None, b12x=b12x)
            if capture_aux
            else residual.new_empty((0, residual.shape[-1]))
        )
        return residual, post.unsqueeze(-1), comb, y, next_pre, aux
    if _split_projection(residual, residual.shape[1] * residual.shape[2], norm_weight):
        # SM80 prefill: the stock post kernel, then the split-k projection
        # and the stock epilogue.
        assert norm_weight is not None
        residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        return residual, *_mhc_pre_split(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix,
            None,
            norm_weight,
            norm_eps,
            capture_aux,
        )
    if can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux):
        assert pre_mix is not None and norm_weight is not None
        args = (
            x,
            residual,
            pre_mix,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_post_mult_value,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        outputs = None
        if boundary_weight is not None:
            # Decode-size batches: one CuTe DSL kernel, the same TF32 products.
            outputs = _mhc_shifted_post_pre_boundary()(
                *args[:5], boundary_weight, *args[6:]
            )
        if outputs is None:
            outputs = mhc_shifted_post_pre_deep_gemm(*args)
        return *outputs, x.new_empty(0, x.shape[1])

    return mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        capture_aux=capture_aux,
    )


@triton.jit
def _hc_expand_kernel(
    x,
    out,
    numel,
    H: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    row = index // H
    col = index % H
    value = tl.load(x + row * STRIDE_M + col * STRIDE_H, mask=mask, other=0)
    for stream in tl.static_range(N):
        tl.store(out + (row * N + stream) * H + col, value, mask=mask)


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """`x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()`, reading
    `x` once instead of once per copy."""
    m, h = x.shape
    out = x.new_empty(m, hc_mult, h)
    if out.numel():
        _hc_expand_kernel[(triton.cdiv(m * h, 1024),)](
            x, out, m * h, h, hc_mult, *x.stride(), BLOCK=1024, num_warps=4
        )
    return out
