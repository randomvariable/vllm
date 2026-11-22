# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed as dist

from vllm.v1.worker import b12x_startup


@pytest.fixture
def make_coordinator(monkeypatch):
    monkeypatch.delenv("B12X_PREPARATION_TRACE_DIR", raising=False)
    store = dist.HashStore()
    store.set_timeout(timedelta(seconds=3))

    def make(rank, ranks, session=None):
        # Each worker has its own stage counter but shares the underlying store.
        monkeypatch.setattr(
            b12x_startup, "_get_control_group", lambda _: SimpleNamespace(store=store)
        )
        return b12x_startup.B12xPreparationCoordinator(
            session,
            [] if session is None else [(("request",), True)],
            global_rank=rank,
            world_group=SimpleNamespace(ranks=ranks),
        )

    return make


@pytest.mark.parametrize(("rank", "missing_key"), [(0, "1"), (1, "decision")])
def test_preparation_missing_participant(
    make_coordinator, monkeypatch, rank, missing_key
):
    monkeypatch.setenv("VLLM_B12X_PREPARATION_CONTROL_TIMEOUT_SECONDS", "1")
    coordinator = make_coordinator(rank, (0, 1))
    with pytest.raises(RuntimeError, match=rf"round-0/{missing_key}.*deadline 1s"):
        coordinator.advance()


def test_preparation_authorizes_distinct_dcp_tuning_keys(make_coordinator, monkeypatch):
    monkeypatch.setenv("VLLM_B12X_PREPARATION_CONTROL_TIMEOUT_SECONDS", "3")
    monkeypatch.setitem(
        sys.modules,
        "b12x.preparation",
        SimpleNamespace(TuningRequirement=Mock(side_effect=lambda *args: args)),
    )
    ranks = (0, 1, 2, 3)
    jobs = []
    coordinators = []
    for rank in ranks:
        progress = SimpleNamespace(
            pending_compilation=False,
            ready_collectives=(),
            ready_tuning=(
                SimpleNamespace(
                    key=f"qsa-dcp-{rank % 2}",
                    ranks=(rank % 2, rank % 2 + 2),
                    assignment=SimpleNamespace(
                        to_dict=lambda rank=rank: {"winner": rank}
                    ),
                    latency_us=float(4 - rank),
                    candidate_index=rank,
                    rejected_count=0,
                    cute_programs=(),
                ),
            ),
            done=False,
        )
        job = Mock()
        job.advance.side_effect = [
            progress,
            SimpleNamespace(
                pending_compilation=False,
                ready_collectives=(),
                ready_tuning=(),
                done=True,
            ),
        ]
        session = Mock()
        session.begin.return_value = job
        jobs.append(job)
        coordinators.append(make_coordinator(rank, ranks, session))

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(
            result["error"] is None
            for result in pool.map(lambda c: c.advance(), coordinators)
        )
        assert all(
            result["done"] for result in pool.map(lambda c: c.advance(), coordinators)
        )
    for rank, job in enumerate(jobs):
        [(key, cohort, assignment, *_)] = job.advance.call_args.kwargs["tuning"]
        assert key == f"qsa-dcp-{rank % 2}"
        assert cohort == (rank % 2, rank % 2 + 2)
        assert assignment == {"winner": rank % 2 + 2}
