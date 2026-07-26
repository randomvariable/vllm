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
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _signature(*names: str) -> inspect.Signature:
    return inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )


def test_probe_requires_exact_0302_plan_owner_bind_contract(monkeypatch):
    integration = ModuleType("b12x.integration")
    tp_moe = ModuleType("b12x.integration.tp_moe")
    integration.__dict__["B12XFP4ExpertWeights"] = Mock(
        __signature__=_signature(
            "plan", "w1_fp4", "w1_blockscale", "w2_fp4", "w2_blockscale"
        )
    )
    integration.__dict__["plan_b12x_fp4_moe_weights"] = Mock(
        __signature__=_signature(
            "quant_modes", "num_experts", "hidden_size", "intermediate_size"
        )
    )
    integration.__dict__["prepare_b12x_fp4_moe_weights"] = Mock(
        __signature__=_signature("plan")
    )
    tp_moe.__dict__["TPMoEScratchCaps"] = Mock(
        __signature__=_signature("weight_plan", "max_tokens")
    )
    tp_moe.__dict__["TPMoEScratchPlan"] = type(
        "TPMoEScratchPlan",
        (),
        {"bind": Mock(__signature__=_signature("self", "experts", "scratch"))},
    )
    monkeypatch.setattr(b12x.importlib.metadata, "version", lambda _: "0.30.2")
    monkeypatch.setattr(
        b12x.importlib,
        "import_module",
        lambda name: tp_moe if name.endswith("tp_moe") else integration,
    )

    assert b12x.has_b12x_moe()
    tp_moe.__dict__["TPMoEScratchPlan"].bind.__signature__ = _signature(
        "self", "scratch"
    )
    assert not b12x.has_b12x_moe()


def test_probe_rejects_wrong_b12x_version(monkeypatch):
    monkeypatch.setattr(b12x.importlib.metadata, "version", lambda _: "0.30.1")
    assert not b12x.has_b12x_moe()


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

    FakeB12xExperts.is_supported_config = Mock(
        return_value=(False, "b12x 0.30.2 is unavailable")
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
    method.moe = SimpleNamespace()
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


def test_preparation_retains_owner_and_plans_persistent_scratch(monkeypatch):
    weight_plan = SimpleNamespace()
    owner = SimpleNamespace()
    scratch_plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((3,), torch.float32), ((2,), torch.int32))
    )
    plan_weights = Mock(return_value=weight_plan)
    prepare_weights = Mock(return_value=owner)
    plan_scratch = Mock(return_value=scratch_plan)
    caps = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(b12x, "_plan_b12x_fp4_moe_weights", plan_weights)
    monkeypatch.setattr(b12x, "_prepare_b12x_fp4_moe_weights", prepare_weights)

    tp_moe = ModuleType("b12x.integration.tp_moe")
    tp_moe.__dict__["TPMoEScratchCaps"] = caps
    tp_moe.__dict__["plan_tp_moe_scratch"] = plan_scratch
    monkeypatch.setitem(sys.modules, "b12x.integration.tp_moe", tp_moe)

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
    experts._scratch = ()
    experts._released_w4a16_source_scales = False
    experts._unit_scale_by_device = {}
    w1 = torch.ones(2, 8, 2)
    w2 = torch.ones(2, 4, 4)

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
    assert tuple(t.shape for t in experts._scratch) == ((3,), (2,))


def test_b12x_rejects_repeated_weight_preparation():
    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._experts = object()

    with pytest.raises(RuntimeError, match="rebuild the engine"):
        experts.process_weights_after_loading(torch.nn.Module())


def test_apply_binds_owner_without_planning_or_allocation(monkeypatch):
    binding = object()
    scratch_plan = SimpleNamespace(bind=Mock(return_value=binding))
    run = Mock()
    tp_moe = ModuleType("b12x.integration.tp_moe")
    tp_moe.__dict__["b12x_moe_fp4"] = run
    monkeypatch.setitem(sys.modules, "b12x.integration.tp_moe", tp_moe)

    experts = cast(Any, object.__new__(b12x.B12xExperts))
    experts._experts = owner = object()
    experts._scratch_plan = scratch_plan
    experts._scratch = (torch.empty(1),)
    hidden = torch.empty(2, 4)
    output = torch.empty_like(hidden)
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_ids = torch.empty(2, 2, dtype=torch.int32)

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
        None,
        None,
        False,
    )

    assert scratch_plan.bind.call_args.kwargs["experts"] is owner
    assert scratch_plan.bind.call_args.kwargs["scratch"] is experts._scratch
    run.assert_called_once_with(binding=binding)
