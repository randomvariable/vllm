# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark drafter detection for DeepSeek-V4 checkpoints.

DeepSeek-V4-Flash-0731 and DeepSeek-V4-Pro-0813 carry DSpark drafters in their
``mtp.*`` tensors, but their repo names contain no ``dspark`` and their
``architectures`` name the target model. Name- and architecture-based detection
therefore routes them to ``DeepSeekV4MTPModel``, which cannot load those
weights. ``dspark_target_layer_ids`` is the discriminator: it appears only on
DSpark checkpoints, while ``num_nextn_predict_layers`` appears on every
DeepSeek-V4 config.
"""

from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.config.speculative import (
    _is_deepseek_v4_dspark,
    _is_dspark_draft,
    reject_mtp_on_dspark,
)

# Repo names that must not be relied on: no substring spells "dspark".
plain_names = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_/."
    ),
    min_size=1,
    max_size=40,
).filter(lambda name: "dspark" not in name.lower())

layer_ids = st.lists(st.integers(min_value=0, max_value=127), min_size=1, max_size=8)


def _config(**kwargs) -> SimpleNamespace:
    kwargs.setdefault("architectures", [])
    return SimpleNamespace(**kwargs)


@given(name=plain_names, ids=layer_ids, n_predict=st.integers(1, 8))
@settings(max_examples=60, deadline=None)
def test_raw_deepseek_v4_dspark_is_detected(name, ids, n_predict):
    """The pre-override config shape: model_type still ``deepseek_v4``."""
    hf_config = _config(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_target_layer_ids=ids,
        num_nextn_predict_layers=n_predict,
    )
    assert _is_deepseek_v4_dspark(hf_config)
    assert _is_dspark_draft(name, hf_config)


@given(name=plain_names, ids=layer_ids)
@settings(max_examples=60, deadline=None)
def test_overridden_mtp_shape_is_detected(name, ids):
    """After hf_config_override: model_type ``deepseek_mtp``, MTP architecture."""
    hf_config = _config(
        model_type="deepseek_mtp",
        architectures=["DeepSeekV4MTPModel"],
        dspark_target_layer_ids=ids,
    )
    assert _is_deepseek_v4_dspark(hf_config)
    assert _is_dspark_draft(name, hf_config)


@given(
    name=plain_names,
    n_predict=st.integers(1, 8),
    empty=st.sampled_from([None, [], ()]),
)
@settings(max_examples=60, deadline=None)
def test_genuine_mtp_head_is_not_detected(name, n_predict, empty):
    """A real MTP checkpoint has no dspark_target_layer_ids to key off."""
    hf_config = _config(
        model_type="deepseek_mtp",
        architectures=["DeepSeekV4MTPModel"],
        num_nextn_predict_layers=n_predict,
    )
    assert not _is_deepseek_v4_dspark(hf_config)
    assert not _is_dspark_draft(name, hf_config)

    hf_config.dspark_target_layer_ids = empty
    assert not _is_deepseek_v4_dspark(hf_config)
    assert not _is_dspark_draft(name, hf_config)


@given(name=plain_names, ids=layer_ids, model_type=st.sampled_from(["llama", "gemma3"]))
@settings(max_examples=60, deadline=None)
def test_dspark_keys_on_unrelated_model_type_are_ignored(name, ids, model_type):
    """The keys only mean DSpark on a DeepSeek-V4 / DeepSeek-V4-MTP config."""
    hf_config = _config(
        model_type=model_type,
        architectures=["LlamaForCausalLM"],
        dspark_target_layer_ids=ids,
    )
    assert not _is_deepseek_v4_dspark(hf_config)
    assert not _is_dspark_draft(name, hf_config)


@given(name=plain_names, model_type=st.sampled_from(["qwen3", "llama", "deepseek_v4"]))
@settings(max_examples=60, deadline=None)
def test_dspark_draft_model_keeps_qwen3_guard(name, model_type):
    """``DSparkDraftModel`` alone is only DSpark on a qwen3 config.

    The fork synthesises this architecture for DeepSeek-V4 DSpark after
    detection, so accepting it unconditionally would make detection depend on
    its own output.
    """
    hf_config = _config(model_type=model_type, architectures=["DSparkDraftModel"])
    assert _is_dspark_draft(name, hf_config) == (model_type == "qwen3")


@given(
    arch=st.sampled_from(["Qwen3DSparkModel", "Gemma4DSparkModel"]),
    name=plain_names,
    model_type=st.sampled_from(["qwen3", "gemma4", "llama"]),
)
@settings(max_examples=40, deadline=None)
def test_self_declaring_dspark_architectures_are_detected(arch, name, model_type):
    hf_config = _config(model_type=model_type, architectures=[arch])
    assert _is_dspark_draft(name, hf_config)


@given(
    prefix=plain_names,
    suffix=plain_names,
    spelling=st.sampled_from(["dspark", "DSpark", "DSPARK", "dSpArK"]),
)
@settings(max_examples=40, deadline=None)
def test_name_fallback_survives_any_casing(prefix, suffix, spelling):
    hf_config = _config(model_type="qwen3")
    assert _is_dspark_draft(f"{prefix}{spelling}{suffix}", hf_config)


def test_missing_architectures_attribute_is_tolerated():
    """Sub-configs of multi-modal models can lack ``architectures`` entirely."""
    hf_config = SimpleNamespace(model_type="deepseek_v4", dspark_target_layer_ids=[43])
    assert _is_deepseek_v4_dspark(hf_config)
    assert _is_dspark_draft("some/model", hf_config)


@given(
    name=plain_names,
    ids=layer_ids,
    block_size=st.integers(min_value=2, max_value=16),
    model_type=st.sampled_from(["deepseek_v4", "deepseek_mtp"]),
)
@settings(max_examples=40, deadline=None)
def test_explicit_mtp_on_dspark_checkpoint_is_rejected(
    name, ids, block_size, model_type
):
    """``method='mtp'`` must fail with guidance, not a weight-loader KeyError."""
    hf_config = _config(
        model_type=model_type,
        architectures=["DeepSeekV4MTPModel"],
        dspark_target_layer_ids=ids,
        dspark_block_size=block_size,
    )
    with pytest.raises(ValueError) as excinfo:
        reject_mtp_on_dspark(method="mtp", model=name, hf_config=hf_config)
    message = str(excinfo.value)
    assert name in message
    assert "dspark" in message
    assert str(block_size) in message


@given(
    method=st.sampled_from(["dspark", "eagle", "ngram", "dflash", None]),
    ids=layer_ids,
)
@settings(max_examples=40, deadline=None)
def test_non_mtp_methods_pass_through_on_dspark_checkpoint(method, ids):
    hf_config = _config(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_target_layer_ids=ids,
    )
    reject_mtp_on_dspark(method=method, model="some/model", hf_config=hf_config)


@given(name=plain_names, n_predict=st.integers(1, 8))
@settings(max_examples=40, deadline=None)
def test_mtp_on_genuine_mtp_head_passes_through(name, n_predict):
    hf_config = _config(
        model_type="deepseek_mtp",
        architectures=["DeepSeekV4MTPModel"],
        num_nextn_predict_layers=n_predict,
    )
    reject_mtp_on_dspark(method="mtp", model=name, hf_config=hf_config)
