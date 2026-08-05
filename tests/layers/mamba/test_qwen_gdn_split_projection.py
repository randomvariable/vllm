# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the config-gated split-projection path in
``QwenGatedDeltaNetAttention``.

The split path (``create_in_proj_qkvz=False`` / ``create_in_proj_ba=False``)
materializes the qkvz / ba projections as four independent layers
(``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``) instead of the
fused ``in_proj_qkvz`` / ``in_proj_ba`` pair, so GGUF models with mixed
per-tensor quantization can keep each projection quantized independently.

These tests are CPU-only and do not exercise any GPU/CPU GDN kernels: they
check ``__init__`` wiring, ``_input_projection`` output shapes, and numerical
equivalence of the split path against the merged path with replicated weights.

Heavy vLLM machinery (distributed init, real ``VllmConfig``, platform
dispatch, CPU-op registration) is mocked out so the test runs without a GPU or
a compiled CPU GDN kernel.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# Heavy imports are deferred so module collection does not hard-fail when the
# surrounding vLLM install is incomplete; each test re-skips if needed.
try:
    from vllm.config.compilation import CompilationConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"vLLM GDN layers unavailable in this environment: {exc!r}",
        allow_module_level=True,
    )


# -- Fixtures --------------------------------------------------------------

HIDDEN = 64
NUM_K_HEADS = 4
NUM_V_HEADS = 4
HEAD_K = 16
HEAD_V = 16
CONV_K = 4


def _make_hf_config(*, split_qkvz: bool, split_ba: bool) -> SimpleNamespace:
    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_num_key_heads=NUM_K_HEADS,
        linear_num_value_heads=NUM_V_HEADS,
        linear_key_head_dim=HEAD_K,
        linear_value_head_dim=HEAD_V,
        linear_conv_kernel_dim=CONV_K,
        # Fork-specific toggles under test.
        create_in_proj_qkvz=split_qkvz,
        create_in_proj_ba=split_ba,
    )
    return cfg


def _make_vllm_config_mock() -> MagicMock:
    """A MagicMock vllm_config whose accessed attributes satisfy the GDN
    base/child constructors. Only the attributes actually read by
    ``QwenGatedDeltaNetAttention.__init__`` are pinned; everything else
    returns a MagicMock lazily.
    """
    vc = MagicMock(name="vllm_config")
    vc.quant_config = None  # unquantized -> plain Linear layers
    vc.speculative_config = None
    vc.model_config.dtype = torch.float32
    # cache_config dtype attrs are read by MambaStateDtypeCalculator; "auto"
    # keeps the path cheap and dtype-stable.
    vc.cache_config.mamba_cache_dtype = "auto"
    vc.cache_config.mamba_ssm_cache_dtype = "auto"
    # compilation_config must be a real CompilationConfig so the
    # static_forward_context dict assignment in __init__ works.
    vc.compilation_config = CompilationConfig()
    return vc


@pytest.fixture()
def gdn_ctx():
    """Patch distributed TP state and platform dispatch so the layer can be
    constructed on CPU without initializing torch.distributed or importing
    the compiled CPU GDN kernel.
    """
    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.base."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn.base."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        # Force the non-CPU/non-XPU/non-ROCm branch so no platform-specific
        # op registration runs during __init__. We never invoke a forward
        # method here, so the chosen _forward_method is irrelevant.
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform"
        ) as mock_platform,
        set_current_vllm_config(_make_vllm_config_mock()),
    ):
        mock_platform.is_cuda.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_rocm.return_value = False
        mock_platform.current_device.return_value = torch.device("cpu")
        yield


def _make_layer(gdn_ctx, *, split_qkvz: bool, split_ba: bool, prefix: str):
    cfg = _make_hf_config(split_qkvz=split_qkvz, split_ba=split_ba)
    return QwenGatedDeltaNetAttention(
        cfg,
        _make_vllm_config_mock(),
        prefix=prefix,
    )


# -- __init__ wiring -------------------------------------------------------


def test_split_path_creates_four_layers_and_no_merged(gdn_ctx):
    layer = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.0")
    # The four split layers exist...
    assert isinstance(layer.in_proj_qkv, MergedColumnParallelLinear)
    assert isinstance(layer.in_proj_z, ColumnParallelLinear)
    assert isinstance(layer.in_proj_b, ColumnParallelLinear)
    assert isinstance(layer.in_proj_a, ColumnParallelLinear)
    # ...and the merged layers do NOT (plugin relies on this exact contract).
    assert not hasattr(layer, "in_proj_qkvz")
    assert not hasattr(layer, "in_proj_ba")
    # Flags are recorded from the HF config.
    assert layer.create_in_proj_qkvz is False
    assert layer.create_in_proj_ba is False


def test_split_prefixes_match_contract(gdn_ctx):
    layer = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.0")
    # The plugin maps GGUF tensors by these exact prefixes.
    assert layer.in_proj_qkv.prefix == "model.layers.0.in_proj_qkv"
    assert layer.in_proj_z.prefix == "model.layers.0.in_proj_z"
    assert layer.in_proj_b.prefix == "model.layers.0.in_proj_b"
    assert layer.in_proj_a.prefix == "model.layers.0.in_proj_a"


def test_merged_default_path_creates_fused_layers(gdn_ctx):
    layer = _make_layer(
        gdn_ctx, split_qkvz=True, split_ba=True, prefix="model.layers.1"
    )
    assert isinstance(layer.in_proj_qkvz, MergedColumnParallelLinear)
    assert isinstance(layer.in_proj_ba, MergedColumnParallelLinear)
    # Split layers must NOT exist on the merged path.
    assert not hasattr(layer, "in_proj_qkv")
    assert not hasattr(layer, "in_proj_z")
    assert not hasattr(layer, "in_proj_b")
    assert not hasattr(layer, "in_proj_a")
    assert layer.create_in_proj_qkvz is True
    assert layer.create_in_proj_ba is True


def test_qkv_split_has_three_outputs_not_four(gdn_ctx):
    layer = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.2")
    key_dim = NUM_K_HEADS * HEAD_K
    value_dim = NUM_V_HEADS * HEAD_V
    # in_proj_qkv is [q, k, v] (3 outputs); the merged in_proj_qkvz has 4
    # ([q, k, v, z]). TP=1 so output_size == full dimension.
    assert layer.in_proj_qkv.output_size == 2 * key_dim + value_dim
    assert layer.in_proj_z.output_size == value_dim
    assert layer.in_proj_b.output_size == NUM_V_HEADS
    assert layer.in_proj_a.output_size == NUM_V_HEADS


# -- _input_projection shapes ---------------------------------------------


def test_input_projection_split_shapes(gdn_ctx):
    layer = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.3")
    key_dim = NUM_K_HEADS * HEAD_K
    value_dim = NUM_V_HEADS * HEAD_V
    tp = 1
    tokens = 7
    hidden_states = torch.randn(tokens, HIDDEN, dtype=torch.float32)
    mixed_qkvz, ba = layer._input_projection(hidden_states)
    # [q, k, v, z] layout: q+k+v = 2*key_dim+value_dim, plus z=value_dim.
    assert mixed_qkvz.shape == (
        tokens,
        (2 * key_dim + value_dim + value_dim) // tp,
    )
    # [b, a] layout: 2 * num_v_heads.
    assert ba.shape == (tokens, (2 * NUM_V_HEADS) // tp)


# -- Numerical equivalence -------------------------------------------------


def _copy_merged_into_split(merged_layer, split_layer) -> None:
    """Replicate the merged in_proj_qkvz / in_proj_ba weight shards into the
    four split layers so both paths produce identical projections at TP=1.

    Non-interleaved Qwen3.5 layout:
      in_proj_qkvz weight rows = [q (key_dim), k (key_dim), v (value_dim),
                                  z (value_dim)]
      in_proj_ba   weight rows = [b (num_v_heads), a (num_v_heads)]
    """
    key_dim = NUM_K_HEADS * HEAD_K
    value_dim = NUM_V_HEADS * HEAD_V

    qkvz_w = merged_layer.in_proj_qkvz.weight.data
    qkv_end = 2 * key_dim + value_dim
    split_layer.in_proj_qkv.weight.data.copy_(qkvz_w[:qkv_end, :])
    split_layer.in_proj_z.weight.data.copy_(qkvz_w[qkv_end:, :])

    ba_w = merged_layer.in_proj_ba.weight.data
    split_layer.in_proj_b.weight.data.copy_(ba_w[:NUM_V_HEADS, :])
    split_layer.in_proj_a.weight.data.copy_(ba_w[NUM_V_HEADS:, :])


def test_split_path_numerically_equivalent_to_merged(gdn_ctx):
    torch.manual_seed(0)
    merged = _make_layer(gdn_ctx, split_qkvz=True, split_ba=True, prefix="model.layers.10")
    split = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.11")

    # Randomize merged projections, then mirror them into the split layers.
    with torch.no_grad():
        merged.in_proj_qkvz.weight.normal_(mean=0.0, std=0.1)
        merged.in_proj_ba.weight.normal_(mean=0.0, std=0.1)
        _copy_merged_into_split(merged, split)

    tokens = 11
    hidden_states = torch.randn(tokens, HIDDEN, dtype=torch.float32)

    merged_qkvz, merged_ba = merged._input_projection(hidden_states)
    split_qkvz, split_ba = split._input_projection(hidden_states)

    torch.testing.assert_close(split_qkvz, merged_qkvz, rtol=0, atol=0)
    torch.testing.assert_close(split_ba, merged_ba, rtol=0, atol=0)


def test_split_path_concat_order_matches_merged_layout(gdn_ctx):
    """Sanity check that the [q,k,v] + [z] concatenation reproduces the
    merged [q,k,v,z] row ordering exactly (not just an all-close pass).
    """
    torch.manual_seed(1)
    merged = _make_layer(gdn_ctx, split_qkvz=True, split_ba=True, prefix="model.layers.20")
    split = _make_layer(gdn_ctx, split_qkvz=False, split_ba=False, prefix="model.layers.21")

    with torch.no_grad():
        merged.in_proj_qkvz.weight.normal_(mean=0.0, std=0.1)
        merged.in_proj_ba.weight.normal_(mean=0.0, std=0.1)
        _copy_merged_into_split(merged, split)

    hidden_states = torch.randn(3, HIDDEN, dtype=torch.float32)
    # Identical inputs to both _input_projection calls.
    (mqkvz, mba) = merged._input_projection(hidden_states)
    (sqkvz, sba) = split._input_projection(hidden_states)
    assert torch.equal(mqkvz, sqkvz)
    assert torch.equal(mba, sba)
