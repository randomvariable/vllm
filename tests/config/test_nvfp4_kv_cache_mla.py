# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 KV cache admission for MLA models.

``nvfp4_ds_mla`` is the packed MLA NVFP4 record read natively by
B12X_MLA_SPARSE, so it must be admitted for MLA models. The dense ``nvfp4``
layouts have no MLA reader and must stay rejected.
"""

from dataclasses import dataclass

import pytest

from vllm.config import VllmConfig


@dataclass
class _Cache:
    cache_dtype: str


@dataclass
class _Model:
    use_mla: bool


@dataclass
class _Cfg:
    """Only the fields the validator reads."""

    cache_config: _Cache
    model_config: _Model | None


def _validate(cache_dtype: str, use_mla: bool) -> None:
    cfg = _Cfg(_Cache(cache_dtype), _Model(use_mla))
    VllmConfig.validate_nvfp4_kv_cache_with_mla(cfg)


@pytest.mark.parametrize("cache_dtype", ["nvfp4", "nvfp4_4over6"])
def test_dense_nvfp4_rejected_for_mla(cache_dtype: str) -> None:
    with pytest.raises(ValueError, match="not supported with MLA"):
        _validate(cache_dtype, use_mla=True)


def test_nvfp4_ds_mla_admitted_for_mla() -> None:
    _validate("nvfp4_ds_mla", use_mla=True)


@pytest.mark.parametrize("cache_dtype", ["nvfp4", "nvfp4_4over6", "nvfp4_ds_mla"])
def test_nvfp4_admitted_for_non_mla(cache_dtype: str) -> None:
    _validate(cache_dtype, use_mla=False)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_non_nvfp4_unaffected(cache_dtype: str) -> None:
    _validate(cache_dtype, use_mla=True)


def test_missing_model_config_is_noop() -> None:
    VllmConfig.validate_nvfp4_kv_cache_with_mla(_Cfg(_Cache("nvfp4"), None))
