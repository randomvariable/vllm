# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixture adapters for boundary checkpoint capacity planning."""

from types import SimpleNamespace

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _estimate_max_model_len_from_groups,
    _pool_bytes_per_block,
)


def estimate_capacity(groups, num_blocks):
    cache = CacheConfig(block_size=2048, mamba_cache_mode="align")
    cache.kv_cache_layout = "BLHNC"
    config = SimpleNamespace(
        use_request_boundary_checkpoints=True,
        max_in_flight_tokens=8192,
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=4096, max_num_batched_tokens=4096
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(max_model_len=1048576),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        cache_config=cache,
    )
    return _estimate_max_model_len_from_groups(
        DefaultKVCacheConfigBuilder(),
        config,
        groups,
        (num_blocks - 1) * _pool_bytes_per_block(config, groups),
    )
