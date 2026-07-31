# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace
from typing import cast

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import (
    flashinfer_mla_sparse,
    flashmla_sparse,
    indexer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    get_prefill_workspace_size,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_backend_selects_dedicated_impl_and_workspace(
    monkeypatch,
) -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))

    monkeypatch.setattr(
        indexer.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        flashmla_sparse.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(
        flashmla_sparse.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    assert get_max_prefill_buffer_size(cast("VllmConfig", config)) == 800
    assert (
        FlashInferMLASparseSM120Backend.get_impl_cls() is FlashInferMLASparseSM120Impl
    )
    assert FlashInferMLASparseSM120Impl.get_workspace_size(100) == 100 * 2 * 576 * 2
    assert get_prefill_workspace_size(100) == 200

    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    monkeypatch.setattr(
        flashmla_sparse.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    assert get_max_prefill_buffer_size(cast("VllmConfig", config)) == 4000
    assert get_prefill_workspace_size(100) == 500


def test_sm120_capability_does_not_reduce_rocm_indexer_workspace(monkeypatch) -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )

    assert get_max_prefill_buffer_size(cast("VllmConfig", config)) == 4000


def test_workspace_buffer_respects_configured_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        flashinfer_mla_sparse.envs,
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
        100,
    )

    flashinfer_mla_sparse._fi_sparse_workspace = None
    smaller = flashinfer_mla_sparse._get_workspace_buffer(
        torch.device("cpu"), workspace_size=50
    )
    assert smaller.numel() == 100

    flashinfer_mla_sparse._fi_sparse_workspace = None
    larger = flashinfer_mla_sparse._get_workspace_buffer(
        torch.device("cpu"), workspace_size=200
    )
    assert larger.numel() == 200
