# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4 MIXED_PRECISION expert dispatch.

Regression test: NVFP4 re-quantizations of DeepSeek-V4-Flash requantize only
the target expert layers and leave the DSpark draft (``mtp.*``) experts in the
checkpoint's original MXFP4 format. The two formats are not interchangeable --
NVFP4 uses an E4M3 scale over groups of 16 plus a per-tensor ``weight_scale_2``,
while MXFP4 uses an E8M0 scale over groups of 32 with no global scale.

``moe_quant_algo`` is a single model-global string, so dispatching on it alone
bound the draft experts to the NVFP4 method. Their scales then failed to load
(silently -- the expert weight loader returns False rather than raising) and
speculative decoding ran against an unloaded draft MoE.

The checkpoint declares the split positively under ``quantized_layers``. Its
companion ``ignore`` key is *not* usable for this: it means "not requantized",
not "unquantized", so the ignored attention and shared-expert layers still
carry block-FP8 weights.
"""

import pytest

from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

# Target layers requantized to NVFP4. The draft block (layers 43+) is absent.
NUM_TARGET_LAYERS = 43
QUANTIZED_LAYERS = {
    f"layers.{i}.ffn.experts": {"group_size": 16, "quant_algo": "NVFP4"}
    for i in range(NUM_TARGET_LAYERS)
}


def _config(quantized_layers=None) -> DeepseekV4FP8Config:
    cfg = DeepseekV4FP8Config()
    # Bypass lazy hf_config resolution; these are what it would have resolved.
    cfg._resolved_expert_dtype = "fp4"
    cfg._resolved_moe_quant_algo = "NVFP4"
    cfg._resolved_nvfp4_expert_layers = frozenset(quantized_layers or ())
    return cfg


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # Target experts: declared NVFP4.
        ("model.layers.0.ffn.experts", True),
        ("model.layers.42.ffn.experts", True),
        # DSpark draft experts are appended after the target layers and are
        # absent from quantized_layers, so they keep their MXFP4 format.
        ("model.layers.43.ffn.experts", False),
        ("model.layers.44.ffn.experts", False),
        ("model.layers.45.ffn.experts", False),
    ],
)
def test_draft_experts_excluded_from_nvfp4(prefix: str, expected: bool) -> None:
    assert _config(QUANTIZED_LAYERS)._is_nvfp4_expert_layer(prefix) is expected


def test_matches_without_model_prefix() -> None:
    """quantized_layers keys are unrooted; layer prefixes carry a root."""
    cfg = _config(QUANTIZED_LAYERS)
    assert cfg._is_nvfp4_expert_layer("layers.0.ffn.experts")
    assert cfg._is_nvfp4_expert_layer("model.layers.0.ffn.experts")


def test_suffix_match_is_component_aligned() -> None:
    """A declared layer must not match a longer numeric neighbour."""
    cfg = _config({"layers.4.ffn.experts": {}})
    assert cfg._is_nvfp4_expert_layer("model.layers.4.ffn.experts")
    assert not cfg._is_nvfp4_expert_layer("model.layers.40.ffn.experts")
    assert not cfg._is_nvfp4_expert_layer("model.layers.14.ffn.experts")


def test_absent_quantized_layers_stays_model_wide() -> None:
    """Checkpoints that make no per-layer distinction keep prior behaviour."""
    cfg = _config(None)
    for prefix in (
        "model.layers.0.ffn.experts",
        "model.layers.43.ffn.experts",
    ):
        assert cfg._is_nvfp4_expert_layer(prefix)
