# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.core.kv_policy_replay import (
    KVTraceConfig,
    KVTraceStep,
    replay_trace,
    request,
    step,
)

pytestmark = pytest.mark.cpu_test


def _prefix_trace(block_size: int) -> tuple[KVTraceStep, ...]:
    prefix = tuple(token for token in range(3) for _ in range(block_size))
    return (
        step(request("first", prefix + (90,))),
        step(request("replay", prefix + (91,))),
    )


def test_dense_prefix_reuse_beats_no_prefix_baseline() -> None:
    config = KVTraceConfig(block_size=8, num_blocks=12)
    trace = _prefix_trace(config.block_size)

    cached = replay_trace(config, trace)
    uncached = replay_trace(config, trace, skip_reading_prefix_cache=True)

    assert cached.steps[1].requests[0].prefix_hit_tokens == 24
    assert cached.prefix_hit_tokens > uncached.prefix_hit_tokens
    assert cached.recomputed_tokens < uncached.recomputed_tokens
    assert cached.admissions == uncached.admissions == (True, True)


def test_small_pool_pressure_is_deterministic_and_order_sensitive() -> None:
    config = KVTraceConfig(block_size=4, num_blocks=5)
    first = tuple(range(16))
    second = tuple(range(16, 32))
    trace = (step(request("a", first), request("b", second)),)

    result_a = replay_trace(config, trace)
    result_b = replay_trace(config, (step(request("b", second), request("a", first)),))

    assert result_a == replay_trace(config, trace)
    assert result_a.steps[0].free_blocks == 4
    outcomes = {
        request_result.request_id: request_result
        for request_result in result_a.steps[0].requests
    }
    for request_result in result_a.steps[0].requests:
        if request_result.request_id == "a":
            assert request_result.admitted
            assert request_result.recomputed_tokens == 16
            assert request_result.group_block_ids == ((1, 2, 3, 4),)
            assert all(
                ref == 0
                for group in request_result.post_free_group_block_refs
                for _, ref in group
            )
        else:
            assert not request_result.admitted
            assert request_result.recomputed_tokens == 0
            assert request_result.group_block_ids == ((),)
    reverse = {
        request_result.request_id: request_result
        for request_result in result_b.steps[0].requests
    }
    assert reverse["b"].admitted
    assert reverse["b"].recomputed_tokens == 16
    assert reverse["b"].group_block_ids == ((1, 2, 3, 4),)
    assert len(set(reverse["b"].group_block_ids[0])) == 4
    assert all(block_id != 0 for block_id in reverse["b"].group_block_ids[0])
    assert reverse["b"].computed_group_block_ids == ((),)
    assert not reverse["a"].admitted
    assert reverse["a"].recomputed_tokens == 0
    assert reverse["a"].per_group_cache_block_ids == ((),)
    assert reverse["a"].computed_group_block_ids == ((),)
    assert reverse["a"].group_block_ids == ((),)
    assert outcomes["a"].group_block_refs == (((1, 1), (2, 1), (3, 1), (4, 1)),)
    outcomes_a = {
        request.request_id: (request.admitted, request.group_block_ids)
        for request in result_a.steps[0].requests
    }
    outcomes_b = {
        request.request_id: (request.admitted, request.group_block_ids)
        for request in result_b.steps[0].requests
    }
    assert outcomes_a != outcomes_b


def test_same_step_prefix_sharing_releases_each_owner() -> None:
    prefix = tuple(token for token in range(3) for _ in range(4))
    result = replay_trace(
        KVTraceConfig(block_size=4, num_blocks=12),
        (
            step(request("seed", prefix + (90,))),
            step(request("first", prefix + (90,)), request("second", prefix + (91,))),
        ),
    )
    first, second = result.steps[1].requests
    shared = set(first.computed_group_block_ids[0])
    assert shared
    assert set(shared) == set(first.group_block_ids[0]) & set(second.group_block_ids[0])
    assert tuple(
        (block_id, ref)
        for block_id, ref in first.pre_free_group_block_refs[0]
        if block_id in shared
    ) == tuple((block_id, 2) for block_id in shared)
    assert tuple(
        (block_id, ref)
        for block_id, ref in first.post_free_group_block_refs[0]
        if block_id in shared
    ) == tuple(
        (block_id, 1)
        for block_id, ref in first.post_free_group_block_refs[0]
        if block_id in shared
    )
    assert tuple(
        (block_id, ref)
        for block_id, ref in second.post_free_group_block_refs[0]
        if block_id in shared
    ) == tuple(
        (block_id, 0)
        for block_id, ref in second.post_free_group_block_refs[0]
        if block_id in shared
    )


def test_hybrid_full_and_sliding_window_groups_retain_only_window_prefix() -> None:
    config = KVTraceConfig(block_size=8, num_blocks=20, sliding_window_blocks=1)
    tokens = tuple(token for token in range(5) for _ in range(config.block_size))
    trace = (step(request("first", tokens)), step(request("replay", tokens)))

    result = replay_trace(
        KVTraceConfig(
            block_size=8,
            num_blocks=20,
            sliding_window_blocks=1,
            retention_interval_blocks=1,
        ),
        trace,
    )
    wider = replay_trace(
        KVTraceConfig(
            block_size=8,
            num_blocks=20,
            sliding_window_blocks=3,
            retention_interval_blocks=1,
        ),
        trace,
    )

    narrow_request = result.steps[1].requests[0]
    wide_request = wider.steps[1].requests[0]
    assert narrow_request.per_group_cache_block_ids[0] == (1, 2, 3, 4)
    assert narrow_request.per_group_cache_block_ids[1] == (0, 0, 0, 9)
    assert wide_request.per_group_cache_block_ids[1] == (0, 7, 8, 9)
    assert (
        narrow_request.per_group_cache_block_ids[1]
        != wide_request.per_group_cache_block_ids[1]
    )
    assert narrow_request.group_block_ids != wide_request.group_block_ids
    assert len(narrow_request.group_block_ids) == 3
    assert len(wide_request.group_block_ids) == 3
    assert narrow_request.group_block_ids[1] != wide_request.group_block_ids[1]
    assert narrow_request.group_block_ids[2] != wide_request.group_block_ids[2]
    assert (
        narrow_request.computed_group_block_ids[0]
        == wide_request.computed_group_block_ids[0]
    )
    assert narrow_request.group_block_ids[1][:3] == (0, 0, 0)
    assert narrow_request.group_block_ids[1][3] != 0
    assert narrow_request.group_block_ids[2][:3] == (0, 0, 0)
    assert narrow_request.group_block_ids[2][3] != 0
    assert wide_request.group_block_ids[1][0] == 0
    assert all(block_id != 0 for block_id in wide_request.group_block_ids[1][1:4])
    assert wide_request.group_block_ids[2][0] == 0
    assert all(block_id != 0 for block_id in wide_request.group_block_ids[2][1:4])


def test_hybrid_replay_preserves_sparse_shared_prefix_boundary() -> None:
    tokens = tuple(token for token in range(5) for _ in range(8))
    longer = tokens + tuple(99 for _ in range(16))
    trace = (
        step(request("first", tokens), evict_groups=(1, 2)),
        step(request("replay", longer)),
        step(request("probe", tokens[:40] + tuple(100 for _ in range(8)))),
    )
    result = replay_trace(
        KVTraceConfig(
            block_size=8,
            num_blocks=40,
            sliding_window_blocks=1,
            retention_interval_blocks=0,
        ),
        trace,
    )

    replay_request = result.steps[1].requests[0]
    assert replay_request.shared_prefix_boundary == 40
    assert replay_request.recomputed_tokens == 56
    assert replay_request.group_block_ids[0][4] != 0
    assert replay_request.group_block_ids[1][3] != 0

    cleared = replay_trace(
        KVTraceConfig(
            block_size=8,
            num_blocks=40,
            sliding_window_blocks=1,
            retention_interval_blocks=0,
        ),
        trace,
        clear_shared_prefix_boundary=True,
    )
    cleared_request = cleared.steps[1].requests[0]
    cleared_probe = cleared.steps[2].requests[0]
    normal_probe = result.steps[2].requests[0]
    assert cleared_request.shared_prefix_boundary == 0
    assert normal_probe.prefix_hit_tokens == 40
    assert cleared_probe.prefix_hit_tokens == 0
    assert (
        normal_probe.per_group_cache_block_ids[1]
        != cleared_probe.per_group_cache_block_ids[1]
    )
    assert normal_probe.per_group_cache_block_ids[1][-1] != 0
    assert normal_probe.prefix_hit_tokens > cleared_probe.prefix_hit_tokens
    assert normal_probe.recomputed_tokens < cleared_probe.recomputed_tokens
    assert normal_probe.computed_group_block_ids[0] == (1, 2, 3, 4, 5)
    assert cleared_probe.computed_group_block_ids == ((), (), ())
    assert normal_probe.group_block_ids != cleared_probe.group_block_ids


def test_replay_respects_max_model_len() -> None:
    config = KVTraceConfig(block_size=4, num_blocks=8, max_model_len=8)
    result = replay_trace(
        config,
        (
            step(request("exact", tuple(range(8)))),
            step(request("too-long", tuple(range(9)))),
        ),
    )

    exact, too_long = (step.requests[0] for step in result.steps)
    assert exact.admitted
    assert exact.recomputed_tokens == 8
    assert too_long.admitted is False
    assert too_long.recomputed_tokens == 0
    assert too_long.group_block_ids == ((),)
