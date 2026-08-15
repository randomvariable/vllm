# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that flat hf_overrides keys are routed to the config that owns them."""

import logging

import pytest
from transformers import LlamaConfig, LlavaConfig

from vllm.transformers_utils.config import _apply_flat_hf_overrides


@pytest.fixture
def wrapper_config() -> LlavaConfig:
    """A multimodal wrapper config whose text config owns the LM parameters."""
    config = LlavaConfig()
    config.text_config.num_experts_per_tok = 8
    return config


def test_flat_key_only_on_text_config_lands_on_text_config(wrapper_config):
    """A key the wrapper does not have must reach the nested text config."""
    assert not hasattr(wrapper_config, "num_experts_per_tok")

    _apply_flat_hf_overrides(wrapper_config, {"num_experts_per_tok": 4})

    assert wrapper_config.get_text_config().num_experts_per_tok == 4
    assert not hasattr(wrapper_config, "num_experts_per_tok")


def test_flat_key_only_on_top_level_stays_top_level(wrapper_config):
    """Keys owned solely by the wrapper keep their existing behaviour."""
    _apply_flat_hf_overrides(wrapper_config, {"image_token_index": 7})

    assert wrapper_config.image_token_index == 7


def test_flat_key_on_both_prefers_top_level(wrapper_config):
    """Attributes present on both configs resolve at the top level."""
    text_config = wrapper_config.get_text_config()
    assert hasattr(wrapper_config, "architectures")
    assert hasattr(text_config, "architectures")

    _apply_flat_hf_overrides(wrapper_config, {"architectures": ["MyArch"]})

    assert wrapper_config.architectures == ["MyArch"]
    assert text_config.architectures != ["MyArch"]


def test_unresolved_flat_key_warns_and_does_not_raise(wrapper_config, caplog):
    """A key resolving nowhere is applied but named in a warning."""
    with caplog.at_level(logging.WARNING):
        _apply_flat_hf_overrides(wrapper_config, {"bogus_key_xyz": 999})

    assert "bogus_key_xyz" in caplog.text
    assert wrapper_config.bogus_key_xyz == 999


def test_single_level_config_behaviour_unchanged():
    """Plain configs are their own text config, so routing is a no-op."""
    config = LlamaConfig()
    assert config.get_text_config() is config

    _apply_flat_hf_overrides(config, {"hidden_size": 1234})

    assert config.hidden_size == 1234


def test_mixed_keys_route_independently(wrapper_config):
    """Each key is routed on its own, not by the batch it arrives in."""
    _apply_flat_hf_overrides(
        wrapper_config,
        {"num_experts_per_tok": 4, "image_token_index": 7},
    )

    assert wrapper_config.get_text_config().num_experts_per_tok == 4
    assert wrapper_config.image_token_index == 7
