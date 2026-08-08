# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe as b12x
import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as oracle
import vllm.model_executor.layers.quantization.mxfp4 as mxfp4_quant
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _signature(*names: str) -> inspect.Signature:
    return inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )


def _fake_fused_moe() -> ModuleType:
    fused_moe = ModuleType("b12x.moe.fused_moe")
    fused_moe.__dict__["ExpertWeights"] = Mock(
        __signature__=_signature(
            "plan", "w1_fp4", "w1_blockscale", "w2_fp4", "w2_blockscale"
        )
    )
    fused_moe.__dict__["plan_weights"] = Mock(
        __signature__=_signature(
            "quant_modes",
            "source_format",
            "activation",
            "params_dtype",
            "num_experts",
            "hidden_size",
            "intermediate_size",
            "w13_layout",
        )
    )
    fused_moe.__dict__["prepare_weights"] = Mock(
        __signature__=_signature("plan", "params_dtype", "w1_fp4", "w2_fp4")
    )
    fused_moe.__dict__["Caps"] = Mock(
        __signature__=_signature("weight_plan", "max_tokens")
    )
    fused_moe.__dict__["Plan"] = type(
        "Plan",
        (),
        {"bind": Mock(__signature__=_signature("self", "experts", "scratch"))},
    )
    return fused_moe


def test_probe_accepts_planned_fused_moe_api(monkeypatch):
    fused_moe = _fake_fused_moe()
    monkeypatch.setattr(b12x.importlib.metadata, "version", lambda _: "1.1.0")
    monkeypatch.setattr(b12x.importlib, "import_module", lambda _: fused_moe)

    assert b12x.has_b12x_moe()


def test_probe_raises_when_planned_bind_contract_is_missing(monkeypatch):
    fused_moe = _fake_fused_moe()
    monkeypatch.setattr(b12x.importlib.metadata, "version", lambda _: "1.1.0")
    monkeypatch.setattr(b12x.importlib, "import_module", lambda _: fused_moe)
    cast(Any, fused_moe.__dict__["Plan"]).bind.__signature__ = _signature(
        "self", "scratch"
    )

    with pytest.raises(RuntimeError, match="Plan.bind"):
        b12x.has_b12x_moe()


def test_probe_returns_false_when_b12x_absent(monkeypatch):
    def _absent(_):
        raise b12x.importlib.metadata.PackageNotFoundError("b12x")

    monkeypatch.setattr(b12x.importlib.metadata, "version", _absent)
    assert not b12x.has_b12x_moe()


def test_probe_raises_on_pre_rename_b12x(monkeypatch):
    """An old b12x must fail loudly, never silently fall back to a slower MoE."""

    def _no_module(_):
        raise ImportError("No module named 'b12x.moe'")

    monkeypatch.setattr(b12x.importlib.metadata, "version", lambda _: "0.30.2")
    monkeypatch.setattr(b12x.importlib, "import_module", _no_module)

    with pytest.raises(RuntimeError, match="Unsupported b12x version"):
        b12x.has_b12x_moe()


def test_swiglu_oai_is_not_claimed_as_silu():
    assert not b12x.B12xExperts._supports_activation(MoEActivation.SWIGLUOAI)


def test_explicit_b12x_oracle_selection_is_fail_closed(monkeypatch):
    class FakeB12xExperts:
        @classmethod
        def is_supported_config(cls, *args):
            return True, None

    monkeypatch.setattr(
        oracle,
        "backend_to_kernel_cls",
        lambda backend: [FakeB12xExperts]
        if backend == Mxfp4MoeBackend.B12X_MXFP4
        else [],
    )
    config = cast(
        FusedMoEConfig,
        SimpleNamespace(
            moe_backend="b12x",
            moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
        ),
    )

    backend, experts_cls = oracle.select_deepseek_v4_mxfp4_moe_backend(config)
    assert backend == Mxfp4MoeBackend.B12X_MXFP4
    assert experts_cls is FakeB12xExperts

    monkeypatch.setattr(
        FakeB12xExperts,
        "is_supported_config",
        classmethod(lambda _cls, *args: (False, "b12x 0.30.2 is unavailable")),
    )
    with pytest.raises(ValueError, match="b12x 0.30.2 is unavailable"):
        oracle.select_deepseek_v4_mxfp4_moe_backend(config)


def test_b12x_processes_canonical_weights_without_generic_conversion(monkeypatch):
    process_weights = Mock()
    kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(process_weights_after_loading=process_weights)
    )
    make_kernel = Mock(return_value=kernel)
    convert_weights = Mock(side_effect=AssertionError("generic converter called"))
    monkeypatch.setattr(mxfp4_quant, "make_mxfp4_moe_kernel", make_kernel)
    monkeypatch.setattr(
        mxfp4_quant,
        "convert_weight_to_mxfp4_moe_kernel_format",
        convert_weights,
    )

    method = cast(Any, object.__new__(mxfp4_quant.Mxfp4MoEMethod))
    method.mxfp4_backend = Mxfp4MoeBackend.B12X_MXFP4
    method.num_experts = 2
    method.intermediate_size = 32
    method.hidden_size = 32
    method._cache_permute_indices = {}
    method.moe_quant_config = None
    method.moe_kernel = None
    method.is_k3_situ_aiter = False
    # w13 fuses gate and up for the gated SILU activation b12x requires, so
    # the canonical w13_weight below is (2, intermediate_size * 2, 16).
    method.moe = SimpleNamespace(w13_num_shards=2)
    method.experts_cls = object
    method.get_fused_moe_quant_config = Mock(return_value=SimpleNamespace())

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(torch.zeros(2, 64, 16, dtype=torch.uint8), False),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(torch.zeros(2, 32, 16, dtype=torch.uint8), False),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(torch.zeros(2, 64, 1, dtype=torch.uint8), False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(torch.zeros(2, 32, 1, dtype=torch.uint8), False),
    )
    layer._expert_routing_tables = Mock(return_value=None)

    canonical_tensors = dict(layer.named_parameters())
    setup_kernel = Mock(wraps=method._setup_kernel)
    method._setup_kernel = setup_kernel
    method.process_weights_after_loading(layer)

    convert_weights.assert_not_called()
    setup_kernel.assert_called_once()
    make_kernel.assert_called_once()
    process_weights.assert_called_once_with(layer)
    assert dict(layer.named_parameters()).keys() == canonical_tensors.keys()
    assert all(
        tensor is canonical_tensors[name] for name, tensor in layer.named_parameters()
    )


def test_b12x_does_not_support_weight_reload():
    method = cast(Any, object.__new__(mxfp4_quant.Mxfp4MoEMethod))
    method.mxfp4_backend = Mxfp4MoeBackend.B12X_MXFP4

    assert not method.supports_weight_reload()


def test_preparation_retains_owner_and_plans_scratch_without_allocating(monkeypatch):
    weight_plan = SimpleNamespace()
    owner = SimpleNamespace()
    scratch_plan = SimpleNamespace(
        scratch_specs=lambda: (SimpleNamespace(shape=(5,), dtype=torch.uint8),)
    )
    plan_weights = Mock(return_value=weight_plan)
    prepare_weights = Mock(return_value=owner)
    plan_scratch = Mock(return_value=scratch_plan)
    caps = Mock(side_effect=lambda **kwargs: kwargs)
    fused_moe = ModuleType("b12x.moe.fused_moe")
    fused_moe.__dict__["plan_weights"] = plan_weights
    fused_moe.__dict__["prepare_weights"] = prepare_weights
    fused_moe.__dict__["Caps"] = caps
    fused_moe.__dict__["plan"] = plan_scratch
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe", fused_moe)

    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts.moe_config = SimpleNamespace(
        hidden_dim=4,
        intermediate_size_per_partition=8,
        max_num_tokens=16,
        max_capture_size=32,
        dp_size=1,
        experts_per_token=2,
    )
    experts.quant_config = SimpleNamespace(
        gemm1_clamp_limit=None,
        w1_scale=torch.ones(2, 1),
        w2_scale=torch.ones(2, 1),
    )
    experts._experts = None
    experts._weight_plan = None
    experts._scratch_plan = None
    experts._released_w4a16_source_scales = False
    experts._unit_scale_by_device = {}
    w1 = torch.ones(2, 8, 2)
    w2 = torch.ones(2, 4, 4)
    allocate = Mock(side_effect=AssertionError("scratch allocated during preparation"))
    monkeypatch.setattr(b12x.torch, "empty", allocate)

    result = experts._get_or_prepare_fp4_moe_weights(
        w1=w1,
        w2=w2,
        activation=MoEActivation.SILU,
        params_dtype=torch.bfloat16,
    )

    assert result is owner
    assert experts._experts is owner
    assert plan_weights.call_args.kwargs["quant_modes"] == "w4a16"
    assert prepare_weights.call_args.kwargs["plan"] is weight_plan
    assert caps.call_args.kwargs["weight_plan"] is weight_plan
    assert caps.call_args.kwargs["max_tokens"] == 32
    assert experts._scratch_plan is scratch_plan
    assert not hasattr(experts, "_scratch")
    allocate.assert_not_called()


def test_b12x_rejects_repeated_weight_preparation():
    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._experts = object()

    with pytest.raises(RuntimeError, match="rebuild the engine"):
        experts.process_weights_after_loading(torch.nn.Module())


def test_apply_binds_owner_to_modular_workspace_without_allocating(monkeypatch):
    binding = object()
    scratch_plan = SimpleNamespace(
        bind=Mock(return_value=binding),
        scratch_specs=lambda: (SimpleNamespace(shape=(5,), dtype=torch.uint8),),
    )
    run = Mock()
    plan_scratch = Mock(side_effect=AssertionError("scratch planned during apply"))
    fused_moe = ModuleType("b12x.moe.fused_moe")
    fused_moe.__dict__["run"] = run
    fused_moe.__dict__["plan"] = plan_scratch
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe", fused_moe)

    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._experts = owner = object()
    experts._scratch_plan = scratch_plan
    hidden = torch.empty(2, 4)
    output = torch.empty_like(hidden)
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_ids = torch.empty(2, 2, dtype=torch.int32)
    empty_weight = torch.empty(0)
    workspace2 = torch.empty(5, dtype=torch.uint8)
    allocate = Mock(side_effect=AssertionError("scratch allocated during apply"))
    monkeypatch.setattr(b12x.torch, "empty", allocate)

    experts.apply(
        output,
        hidden,
        empty_weight,
        empty_weight,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        2,
        None,
        None,
        None,
        None,
        workspace2,
        None,
        False,
    )

    assert scratch_plan.bind.call_args.kwargs["experts"] is owner
    scratch = scratch_plan.bind.call_args.kwargs["scratch"][0]
    assert scratch.dtype == torch.uint8
    assert scratch.numel() == 5
    assert (
        scratch.untyped_storage().data_ptr() == workspace2.untyped_storage().data_ptr()
    )
    allocate.assert_not_called()
    plan_scratch.assert_not_called()
    run.assert_called_once_with(binding=binding)


def test_workspace_shapes_expose_planned_scratch_arena():
    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._scratch_plan = SimpleNamespace(
        scratch_specs=lambda: (SimpleNamespace(shape=(513,), dtype=torch.uint8),)
    )

    assert experts.workspace_shapes(2, 8, 4, 2, 2, 2, None, MoEActivation.SILU) == (
        (0,),
        (257,),
        (2, 4),
    )


def test_modular_workspace_keeps_output_and_odd_scratch_disjoint(monkeypatch):
    import vllm.v1.worker.workspace as workspace

    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)

    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._scratch_plan = SimpleNamespace(
        scratch_specs=lambda: (SimpleNamespace(shape=(513,), dtype=torch.uint8),)
    )
    experts.workspace_dtype = lambda _: torch.bfloat16
    experts.workspace_shapes = b12x.B12xExperts.workspace_shapes.__get__(experts)
    kernel = cast(Any, object.__new__(FusedMoEKernelModularImpl))
    kernel.fused_experts = experts

    _, scratch_workspace, output = kernel._allocate_buffers(
        torch.bfloat16,
        torch.device("cpu"),
        2,
        2,
        8,
        4,
        2,
        2,
        2,
        None,
        MoEActivation.SILU,
    )
    scratch = scratch_workspace.view(torch.uint8)[:513]
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    scratch_start = scratch.data_ptr()
    scratch_end = scratch_start + scratch.numel()

    assert output_end <= scratch_start or scratch_end <= output_start
    assert scratch.numel() == 513

    _, scratch_again, output_again = kernel._allocate_buffers(
        torch.bfloat16,
        torch.device("cpu"),
        2,
        2,
        8,
        4,
        2,
        2,
        2,
        None,
        MoEActivation.SILU,
    )
    assert scratch_again.data_ptr() == scratch_workspace.data_ptr()
    assert output_again.data_ptr() == output.data_ptr()


def test_distinct_experts_bind_same_modular_workspace(monkeypatch):
    run = Mock()
    monkeypatch.setattr(b12x, "_run_moe", run)

    hidden = torch.empty(2, 4)
    output = torch.empty_like(hidden)
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_ids = torch.empty(2, 2, dtype=torch.int32)
    workspace2 = torch.empty(5, dtype=torch.uint8)
    for _ in range(2):
        experts = cast(Any, object.__new__(b12x.B12xExperts))
        experts._experts = object()
        experts._scratch_plan = SimpleNamespace(
            scratch_specs=lambda: (SimpleNamespace(shape=(5,), dtype=torch.uint8),)
        )

        experts.apply(
            output,
            hidden,
            torch.empty(0),
            torch.empty(0),
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            2,
            None,
            None,
            None,
            None,
            workspace2,
            None,
            False,
        )

    assert all(
        call.kwargs["scratch"][0].untyped_storage().data_ptr()
        == workspace2.untyped_storage().data_ptr()
        for call in run.call_args_list
    )
