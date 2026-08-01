# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for standard FP8 MoE backend selection."""

from typing import cast

import pytest
import torch
from typing_extensions import TypedDict

from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import fp8 as fp8_quant
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


def _make_fp8_moe_config(moe_backend: MoEBackend = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def cutlass_support(monkeypatch):
    """Make kernel support predicates hardware-independent."""
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsFp8,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )

    monkeypatch.setattr(
        CutlassExpertsFp8, "_supports_current_device", staticmethod(lambda: True)
    )
    monkeypatch.setattr(
        TritonExperts, "_supports_current_device", staticmethod(lambda: True)
    )
    monkeypatch.setattr(current_platform, "supports_fp8", lambda: True)


@pytest.mark.usefixtures("cutlass_support")
@pytest.mark.parametrize(
    "weight_key,activation_key",
    [
        (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        (kFp8StaticTensorSym, kFp8DynamicTensorSym),
    ],
)
def test_fp8_explicit_cutlass_supports_standard_schemes(
    weight_key, activation_key
):
    config = _make_fp8_moe_config(moe_backend="cutlass")
    backend, experts_cls = select_fp8_moe_backend(
        config,
        weight_key=weight_key,
        activation_key=activation_key,
        allow_vllm_cutlass=True,
    )

    assert backend == Fp8MoeBackend.VLLM_CUTLASS
    assert experts_cls is not None


@pytest.mark.usefixtures("cutlass_support")
def test_fp8_explicit_cutlass_rejects_block_scheme():
    config = _make_fp8_moe_config(moe_backend="cutlass")
    with pytest.raises(ValueError, match="quantization scheme"):
        select_fp8_moe_backend(
            config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
            allow_vllm_cutlass=True,
        )


def test_fp8_explicit_cutlass_can_be_disabled():
    config = _make_fp8_moe_config(moe_backend="cutlass")
    with pytest.raises(ValueError, match="backend is disabled"):
        select_fp8_moe_backend(
            config,
            weight_key=kFp8StaticTensorSym,
            activation_key=kFp8DynamicTensorSym,
            allow_vllm_cutlass=False,
        )


def test_fp8_static_activation_disables_cutlass():
    config = _make_fp8_moe_config(moe_backend="cutlass")
    with pytest.raises(ValueError, match="backend is disabled"):
        select_fp8_moe_backend(
            config,
            weight_key=kFp8StaticTensorSym,
            activation_key=kFp8StaticTensorSym,
            allow_vllm_cutlass=False,
        )


class _BackendSelection(TypedDict):
    allow_vllm_cutlass: bool


@pytest.mark.parametrize(
    "activation_scheme,weight_block_size,expected_allow",
    [
        ("dynamic", None, True),
        ("static", None, False),
        ("dynamic", [128, 128], False),
    ],
)
def test_fp8_moe_method_computes_cutlass_gate(
    monkeypatch,
    activation_scheme,
    weight_block_size,
    expected_allow,
):
    selections: list[_BackendSelection] = []

    def select_backend(*args, **kwargs):
        selections.append({"allow_vllm_cutlass": kwargs["allow_vllm_cutlass"]})
        return Fp8MoeBackend.TRITON, None

    monkeypatch.setattr(fp8_quant, "select_fp8_moe_backend", select_backend)

    quant_config = fp8_quant.Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
        weight_block_size=weight_block_size,
    )
    layer = cast(RoutedExperts, object.__new__(RoutedExperts))
    layer.moe_config = _make_fp8_moe_config()

    fp8_quant.Fp8MoEMethod(quant_config, layer)

    assert selections == [{"allow_vllm_cutlass": expected_allow}]
