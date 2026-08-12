# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.config import CacheConfig
from vllm.utils.mem_constants import GiB_bytes as GiB
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.utils import (
    bind_kv_cache,
    describe_memory_budget,
    effective_memory_budget,
    request_memory,
)


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


# request_memory resolves the single byte target every GPU-like worker
# profiles against, from whichever memory control is active on CacheConfig.


def _memory_snapshot(free_memory: int, total_memory: int) -> MemorySnapshot:
    return MemorySnapshot(
        free_memory=free_memory,
        total_memory=total_memory,
        device="cpu",
        auto_measure=False,
    )


def test_request_memory_fractional():
    """The fractional control keeps its historical ceil(total * fraction)."""
    snapshot = _memory_snapshot(free_memory=80 * GiB, total_memory=80 * GiB)
    cache_config = CacheConfig(gpu_memory_utilization=0.5)

    assert request_memory(snapshot, cache_config) == math.ceil(80 * GiB * 0.5)


def test_request_memory_absolute():
    """The absolute control is an exact GiB byte target."""
    snapshot = _memory_snapshot(free_memory=100 * GiB, total_memory=120 * GiB)
    cache_config = CacheConfig(gpu_memory_utilization_gb=40.5)

    assert request_memory(snapshot, cache_config) == math.ceil(40.5 * GiB)


def test_request_memory_absolute_rounds_up_partial_bytes():
    """A GiB value that is not a whole number of bytes rounds up, never down."""
    snapshot = _memory_snapshot(free_memory=100 * GiB, total_memory=120 * GiB)
    cache_config = CacheConfig(gpu_memory_utilization_gb=0.1)

    # 0.1 GiB is 107374182.4 bytes: truncation would under-budget by a byte.
    assert request_memory(snapshot, cache_config) == 107374183


def test_request_memory_absolute_is_independent_of_total_memory():
    """Unlike a fraction, the absolute budget does not scale with the device."""
    cache_config = CacheConfig(gpu_memory_utilization_gb=32.0)
    small = _memory_snapshot(free_memory=64 * GiB, total_memory=64 * GiB)
    large = _memory_snapshot(free_memory=64 * GiB, total_memory=512 * GiB)

    assert request_memory(small, cache_config) == request_memory(large, cache_config)
    assert request_memory(small, cache_config) == 32 * GiB


def test_request_memory_insufficient_free_memory_reports_active_budget():
    """The startup free-memory check names the control the user actually set,
    its configured value, and the resolved byte target."""
    snapshot = _memory_snapshot(free_memory=10 * GiB, total_memory=120 * GiB)

    with pytest.raises(ValueError, match=r"--gpu-memory-utilization-gb=40\.0"):
        request_memory(snapshot, CacheConfig(gpu_memory_utilization_gb=40.0))

    with pytest.raises(ValueError, match=r"--gpu-memory-utilization=0\.9"):
        request_memory(snapshot, CacheConfig(gpu_memory_utilization=0.9))


def test_effective_memory_budget_uses_default_fraction():
    """An unset budget resolves through CacheConfig's 0.92 default."""
    total = 80 * GiB

    assert effective_memory_budget(total, CacheConfig()) == math.ceil(total * 0.92)


def test_describe_memory_budget_names_active_control():
    """Diagnostics name the flag the user actually set, and its value."""
    assert (
        describe_memory_budget(CacheConfig(gpu_memory_utilization_gb=40.0))
        == "--gpu-memory-utilization-gb=40.0"
    )
    assert (
        describe_memory_budget(CacheConfig(gpu_memory_utilization=0.5))
        == "--gpu-memory-utilization=0.5"
    )
