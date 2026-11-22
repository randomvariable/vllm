# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed import parallel_state
from vllm.model_executor.warmup import b12x_prepare
from vllm.model_executor.warmup.b12x_prepare import get_b12x_session
from vllm.v1.worker import b12x_startup


def _begin_on_extensible_pool(monkeypatch, stage, *, committable=10746):
    """Begin a b12x stage on a worker whose serving KV cache is extensible.

    The Qwen3.8 smoke geometry: ten KV cache groups of 3008-token blocks and an
    8192-token context, so one QSA prefill prime spans blocks 0..2.
    """
    monkeypatch.setattr(b12x_prepare, "b12x_native_supported", lambda worker: True)
    monkeypatch.setattr(b12x_prepare, "b12x_workload", lambda worker, stage: None)
    monkeypatch.setattr(b12x_prepare, "collect_b12x_units", lambda worker, w: [])
    monkeypatch.setattr(b12x_startup, "B12xPreparationCoordinator", Mock())
    monkeypatch.setattr(parallel_state, "get_world_group", Mock())
    kv_cache = SimpleNamespace(warmup_committable_blocks=committable, commit=Mock())
    runner = SimpleNamespace(
        extensible_kv_cache=kv_cache,
        max_model_len=8192,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(block_size=3008)] * 10
        ),
    )
    worker = SimpleNamespace(
        rank=0,
        model_runner=runner,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_b12x_autotune=False)
        ),
    )
    b12x_prepare.begin_b12x_preparation(worker, stage=stage)
    return kv_cache.commit


@pytest.mark.parametrize("stage", ["state", "bind"])
def test_pool_stages_commit_blocks_primes_address(monkeypatch, stage):
    # Allocation backs only the null blocks; a prime writes blocks 0..2.
    commit = _begin_on_extensible_pool(monkeypatch, stage)
    commit.assert_called_once_with(31)


def test_prime_blocks_stay_within_committable_blocks(monkeypatch):
    commit = _begin_on_extensible_pool(monkeypatch, "bind", committable=8)
    commit.assert_called_once_with(8)


def test_weights_stage_commits_no_kv_blocks(monkeypatch):
    commit = _begin_on_extensible_pool(monkeypatch, "weights")
    commit.assert_not_called()


@pytest.mark.parametrize(
    ("tp_ranks", "dcp_groups", "cohorts"),
    [
        pytest.param(
            (5, 7, 8, 9),
            ((5,), (7,), (8,), (9,)),
            ((5, 7, 8, 9),),
            id="dcp1",
        ),
        pytest.param(
            (0, 1, 2, 3),
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            id="crs804-tp4-dcp2",
        ),
        pytest.param(
            (10, 12, 15, 18, 22, 24),
            ((10, 15, 22), (12, 18, 24)),
            ((10, 12), (15, 18), (22, 24)),
            id="nonconsecutive-tp6-dcp3",
        ),
        pytest.param(
            (4, 5, 6, 7),
            ((0, 1, 2, 3), (4, 5, 8, 9), (6, 7, 10, 11)),
            ((4, 6), (5, 7)),
            id="dcp-spans-pcp",
        ),
    ],
)
def test_b12x_session_shards_tuning_by_dcp_rank(
    monkeypatch, tp_ranks, dcp_groups, cohorts
):
    monkeypatch.delenv("B12X_AUTOTUNE", raising=False)
    session = Mock(autotune=True)
    provider = SimpleNamespace(PreparationSession=Mock(return_value=session))
    monkeypatch.setitem(sys.modules, "b12x.preparation", provider)
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(ranks=tp_ranks)
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="test", dtype="bfloat16"),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=len(tp_ranks),
            pipeline_parallel_size=1,
            decode_context_parallel_size=len(dcp_groups[0]),
        ),
        speculative_config=None,
        kernel_config=SimpleNamespace(enable_b12x_autotune=True),
    )
    for cohort in cohorts:
        for rank in cohort:
            dcp_ranks = next(group for group in dcp_groups if rank in group)
            dcp_group = SimpleNamespace(
                ranks=dcp_ranks,
                group_ranks=dcp_groups,
                rank_in_group=dcp_ranks.index(rank),
                world_size=len(dcp_ranks),
            )
            monkeypatch.setattr(
                parallel_state, "get_dcp_group", lambda group=dcp_group: group
            )
            worker = SimpleNamespace(rank=rank, vllm_config=config, device="cpu")
            assert get_b12x_session(worker) is session
            session.configure_tuning_shard.assert_called_once_with(rank, cohort)
            session.configure_tuning_shard.reset_mock()
