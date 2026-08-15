# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, deterministic replay harness for KV-cache policy tests."""

from dataclasses import dataclass, replace

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm import envs
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash


@dataclass(frozen=True)
class KVTraceConfig:
    """CPU-only cache configuration used by a replay."""

    block_size: int
    num_blocks: int
    max_model_len: int = 8192
    sliding_window_blocks: int | None = None
    retention_interval_blocks: int | None = None

    @property
    def scheduler_block_size(self) -> int:
        return self.block_size


@dataclass(frozen=True)
class KVTraceRequest:
    """Immutable request description for one replay step."""

    request_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class KVTraceStep:
    """One ordered batch of requests in a trace."""

    requests: tuple[KVTraceRequest, ...]
    evict_groups: tuple[int, ...] = ()


@dataclass(frozen=True)
class KVReplayRequest:
    """Request-owned metrics, retaining production KV group structure."""

    request_id: str
    requested_tokens: int
    admitted: bool
    prefix_hit_tokens: int
    recomputed_tokens: int
    shared_prefix_boundary: int
    per_group_cache_block_ids: tuple[tuple[int, ...], ...]
    computed_group_block_ids: tuple[tuple[int, ...], ...]
    group_block_ids: tuple[tuple[int, ...], ...]
    group_block_refs: tuple[tuple[tuple[int, int], ...], ...]
    pre_free_group_block_refs: tuple[tuple[tuple[int, int], ...], ...]
    post_free_group_block_refs: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class KVReplayStep:
    requests: tuple[KVReplayRequest, ...]
    free_blocks: int
    cache_events: int


@dataclass(frozen=True)
class KVReplayResult:
    """Stable metrics emitted by a trace replay."""

    steps: tuple[KVReplayStep, ...]

    @property
    def admissions(self) -> tuple[bool, ...]:
        return tuple(
            request.admitted for step in self.steps for request in step.requests
        )

    @property
    def prefix_hit_tokens(self) -> int:
        return sum(
            request.prefix_hit_tokens
            for step in self.steps
            for request in step.requests
        )

    @property
    def recomputed_tokens(self) -> int:
        return sum(
            request.recomputed_tokens
            for step in self.steps
            for request in step.requests
        )


def _make_manager(config: KVTraceConfig) -> KVCacheManager:
    from tests.v1.core.test_prefix_caching import make_kv_cache_config

    if config.sliding_window_blocks is None:
        kv_cache_config = make_kv_cache_config(config.block_size, config.num_blocks)
    else:
        from tests.v1.core.test_prefix_caching import make_kv_cache_config_hybrid_model

        kv_cache_config = make_kv_cache_config_hybrid_model(
            config.block_size, config.num_blocks, config.sliding_window_blocks
        )
    env_name = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"
    missing = object()
    old_retention_interval = envs.__dict__.get(env_name, missing)
    envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = (
        None
        if config.retention_interval_blocks is None
        else config.retention_interval_blocks * config.block_size
    )
    try:
        return make_kv_cache_manager(
            kv_cache_config,
            max_model_len=config.max_model_len,
            enable_caching=True,
            hash_block_size=config.block_size,
            enable_kv_cache_events=True,
        )
    finally:
        if old_retention_interval is missing:
            delattr(envs, env_name)
        else:
            setattr(envs, env_name, old_retention_interval)


def replay_trace(
    config: KVTraceConfig,
    trace: tuple[KVTraceStep, ...],
    *,
    skip_reading_prefix_cache: bool = False,
    clear_shared_prefix_boundary: bool = False,
) -> KVReplayResult:
    """Replay requests through production allocation and eviction code.

    The no-prefix baseline bypasses lookup only. Production caching and
    eviction remain enabled, keeping allocator behavior comparable.
    """
    init_none_hash(sha256)
    manager = _make_manager(config)
    results: list[KVReplayStep] = []

    for trace_step in trace:
        request_results: list[KVReplayRequest] = []
        allocated_requests = []
        for request_spec in trace_step.requests:
            request = make_request(
                request_spec.request_id,
                list(request_spec.token_ids),
                config.block_size,
                sha256,
            )
            request.skip_reading_prefix_cache = skip_reading_prefix_cache
            empty_groups = tuple(() for _ in range(manager.num_kv_cache_groups))
            if request.num_tokens > config.max_model_len:
                request_results.append(
                    KVReplayRequest(
                        request_id=request.request_id,
                        requested_tokens=request.num_tokens,
                        admitted=False,
                        prefix_hit_tokens=0,
                        recomputed_tokens=0,
                        shared_prefix_boundary=0,
                        per_group_cache_block_ids=empty_groups,
                        computed_group_block_ids=empty_groups,
                        group_block_ids=empty_groups,
                        group_block_refs=tuple(() for _ in empty_groups),
                        pre_free_group_block_refs=tuple(
                            () for _ in empty_groups
                        ),
                        post_free_group_block_refs=tuple(() for _ in empty_groups),
                    )
                )
                continue
            computed_blocks, num_new_computed_tokens, shared_prefix_boundary = (
                manager.get_computed_blocks(request)
            )
            find_per_group = getattr(
                manager.coordinator, "find_longest_cache_hit_per_group", None
            )
            if find_per_group is None:
                per_group_blocks = computed_blocks.blocks
            else:
                per_group_blocks, _ = find_per_group(
                    request.block_hashes, request.num_tokens - 1
                )
            if clear_shared_prefix_boundary:
                shared_prefix_boundary = 0
            request.shared_prefix_boundary = shared_prefix_boundary
            num_new_tokens = request.num_tokens - num_new_computed_tokens
            if num_new_tokens <= 0:
                raise ValueError("replay requires at least one token to allocate")
            blocks = manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=computed_blocks,
            )
            group_block_ids = (
                tuple(
                    tuple(group)
                    for group in manager.get_block_ids(request.request_id)
                )
                if blocks is not None
                else tuple(() for _ in range(manager.num_kv_cache_groups))
            )
            computed_group_block_ids = tuple(
                tuple(block.block_id for block in group if not block.is_null)
                for group in computed_blocks.blocks
            )
            group_block_refs = tuple(
                tuple(
                    (block_id, manager.block_pool.blocks[block_id].ref_cnt)
                    for block_id in group
                    if block_id != 0
                )
                for group in group_block_ids
            )
            if blocks is not None:
                assert len(group_block_ids) == manager.num_kv_cache_groups
                assert all(
                    len(non_null := tuple(block_id for block_id in group if block_id))
                    == len(set(non_null))
                    for group in group_block_ids
                )
                allocated_requests.append(request)
            request_results.append(
                KVReplayRequest(
                    request_id=request.request_id,
                    requested_tokens=request.num_tokens,
                    admitted=blocks is not None,
                    prefix_hit_tokens=(
                        num_new_computed_tokens if blocks is not None else 0
                    ),
                    recomputed_tokens=num_new_tokens if blocks is not None else 0,
                    shared_prefix_boundary=(
                        shared_prefix_boundary if blocks is not None else 0
                    ),
                    per_group_cache_block_ids=tuple(
                        tuple(block.block_id for block in group)
                        for group in per_group_blocks
                    ),
                    computed_group_block_ids=(
                        computed_group_block_ids
                        if blocks is not None
                        else tuple(() for _ in range(manager.num_kv_cache_groups))
                    ),
                    group_block_ids=group_block_ids,
                    group_block_refs=group_block_refs,
                    pre_free_group_block_refs=tuple(
                        () for _ in range(manager.num_kv_cache_groups)
                    ),
                    post_free_group_block_refs=tuple(
                        () for _ in range(manager.num_kv_cache_groups)
                    ),
                )
            )

        live_results = {
            result.request_id: result
            for result in request_results
            if result.admitted
        }
        for result in live_results.values():
            ids = [
                block_id
                for group in result.group_block_ids
                for block_id in group
                if block_id != 0
            ]
            assert len(ids) == len(set(ids))

        live_block_owners: dict[int, list[str]] = {}
        for result in request_results:
            if not result.admitted:
                continue
            for group in result.group_block_ids:
                for block_id in group:
                    if block_id:
                        live_block_owners.setdefault(block_id, []).append(
                            result.request_id
                        )
        for block_id, owners in live_block_owners.items():
            if len(owners) > 1:
                assert manager.block_pool.blocks[block_id].ref_cnt >= len(owners)

        for request in allocated_requests:
            pre_free_refs = tuple(
                tuple(
                    (
                        manager.block_pool.blocks[block_id],
                        block_id,
                        manager.block_pool.blocks[block_id].ref_cnt,
                    )
                    for block_id in group
                    if block_id
                )
                for group in manager.get_block_ids(request.request_id)
            )
            manager.free(request)
            assert all(
                not group for group in manager.get_block_ids(request.request_id)
            )
            for group in pre_free_refs:
                for block, block_id, before_ref in group:
                    assert block.ref_cnt == before_ref - 1

            request_result = next(
                result
                for result in request_results
                if result.request_id == request.request_id
            )
            index = request_results.index(request_result)
            request_results[index] = replace(
                request_result,
                pre_free_group_block_refs=tuple(
                    tuple(
                        (block_id, before_ref)
                        for _, block_id, before_ref in group
                    )
                    for group in pre_free_refs
                ),
                post_free_group_block_refs=tuple(
                    tuple((block_id, block.ref_cnt) for block, block_id, _ in group)
                    for group in pre_free_refs
                ),
            )

        for group_index in trace_step.evict_groups:
            block_ids = {
                block_id
                for result in request_results
                for block_id in result.group_block_ids[group_index]
                if block_id
            }
            manager.evict_blocks(block_ids)

        results.append(
            KVReplayStep(
                requests=tuple(request_results),
                free_blocks=manager.block_pool.get_num_free_blocks(),
                cache_events=len(manager.take_events()),
            )
        )
    return KVReplayResult(tuple(results))


def request(request_id: str, token_ids: tuple[int, ...]) -> KVTraceRequest:
    """Convenience constructor that keeps trace call sites compact."""
    return KVTraceRequest(request_id, token_ids)


def step(
    *requests: KVTraceRequest, evict_groups: tuple[int, ...] = ()
) -> KVTraceStep:
    """Convenience constructor for an ordered replay step."""
    return KVTraceStep(tuple(requests), evict_groups)
