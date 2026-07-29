# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import CUDAGraphMode
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(
    config_hash="abc123",
    free_memory=78 * GiB_bytes,
    kv_bytes=None,
    gpu_memory_utilization_gb=None,
) -> Any:
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=kv_bytes,
            gpu_memory_utilization_gb=gpu_memory_utilization_gb,
        ),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_startup_plan_disabled_under_absolute_budget(plan_env):
    """An absolute GiB budget is a whole-engine target enforced by profiling.

    Applying a plan converts the boot to the explicit-KV fast path, which
    skips profiling and would bypass that enforcement, so plans are neither
    applied nor saved while the absolute budget is active.
    """
    absolute = _plan_worker(gpu_memory_utilization_gb=40.0)
    maybe_save_startup_plan(absolute, 50 * GiB_bytes)
    maybe_apply_startup_plan(absolute)
    assert absolute.cache_config.kv_cache_memory_bytes is None

    # A fractional boot with the same fingerprint must not see a saved plan
    # from the absolute boot either.
    fractional = _plan_worker()
    maybe_apply_startup_plan(fractional)
    assert fractional.cache_config.kv_cache_memory_bytes is None

    # ... and the fractional path itself is unchanged.
    maybe_save_startup_plan(fractional, 50 * GiB_bytes)
    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes


class TestDetermineAvailableMemoryBudget:
    """`Worker.determine_available_memory` must subtract the same non-KV terms
    from whichever memory budget is active, so the absolute GiB target really
    is a total engine-resident target."""

    @staticmethod
    def _worker(
        requested_memory: int,
        non_kv_cache_memory: int,
        cudagraph_estimate: int,
        gpu_memory_utilization_gb: float | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> Any:
        from vllm.v1.worker.gpu_worker import Worker

        # Bound as Any: the profiling collaborators below are stand-ins, not
        # real config/runner objects.
        worker: Any = Worker.__new__(Worker)
        worker.requested_memory = requested_memory
        worker.init_snapshot = SimpleNamespace(
            free_memory=200 * GiB_bytes,
            total_memory=256 * GiB_bytes,
        )
        worker.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=gpu_memory_utilization,
            gpu_memory_utilization_gb=gpu_memory_utilization_gb,
        )
        worker.model_config = SimpleNamespace(multimodal_config=None)
        worker.parallel_config = SimpleNamespace(_api_process_count=1)
        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = 0
        worker.model_runner.profile_cudagraph_memory.return_value = cudagraph_estimate
        worker.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
        )
        worker._profile_result = SimpleNamespace(
            non_kv_cache_memory=non_kv_cache_memory,
            transient_peak_headroom=0,
            total_consumed=non_kv_cache_memory,
            after_profile=SimpleNamespace(free_memory=150 * GiB_bytes),
        )
        return worker

    @staticmethod
    @contextmanager
    def _patched(worker: Any):
        from vllm.v1.worker import gpu_worker as gpu_worker_mod

        @contextmanager
        def fake_memory_profiling(*args, **kwargs):
            yield worker._profile_result

        with (
            patch.object(gpu_worker_mod, "maybe_apply_startup_plan", lambda w: None),
            patch.object(gpu_worker_mod, "memory_profiling", fake_memory_profiling),
            patch.object(
                gpu_worker_mod,
                "reserve_mm_ipc_gpu_memory",
                lambda available, *args, **kwargs: available,
            ),
            patch.object(
                gpu_worker_mod.current_platform, "is_cuda_alike", lambda: True
            ),
        ):
            yield

    def test_absolute_budget_subtracts_non_kv_and_graphs(self):
        """Available KV = absolute target - non-KV - CUDA graph estimate."""
        worker = self._worker(
            requested_memory=40 * GiB_bytes,
            non_kv_cache_memory=12 * GiB_bytes,
            cudagraph_estimate=2 * GiB_bytes,
            gpu_memory_utilization_gb=40.0,
        )
        with self._patched(worker):
            available = worker.determine_available_memory()

        assert available == 40 * GiB_bytes - 12 * GiB_bytes - 2 * GiB_bytes

    def test_absolute_budget_subtracts_graphs_even_when_estimator_disabled(
        self, monkeypatch
    ):
        """In absolute mode the graph estimate is always accounted for, so the
        env opt-out cannot silently make the total target incomplete."""
        monkeypatch.setattr("vllm.envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", False)
        worker = self._worker(
            requested_memory=40 * GiB_bytes,
            non_kv_cache_memory=12 * GiB_bytes,
            cudagraph_estimate=2 * GiB_bytes,
            gpu_memory_utilization_gb=40.0,
        )
        with self._patched(worker):
            available = worker.determine_available_memory()

        assert available == 40 * GiB_bytes - 12 * GiB_bytes - 2 * GiB_bytes

    def test_absolute_budget_fails_closed_on_zero_graph_estimate(self):
        """A graph estimator that reports 0 while capture is active would make
        the absolute budget silently overshoot, so fail closed instead."""
        worker = self._worker(
            requested_memory=40 * GiB_bytes,
            non_kv_cache_memory=12 * GiB_bytes,
            cudagraph_estimate=0,
            gpu_memory_utilization_gb=40.0,
        )
        with self._patched(worker), pytest.raises(ValueError, match="gpu-memory-"):
            worker.determine_available_memory()

    def test_fractional_budget_respects_estimator_opt_out(self, monkeypatch):
        """Fraction mode keeps its existing env-controlled behaviour."""
        monkeypatch.setattr("vllm.envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", False)
        worker = self._worker(
            requested_memory=40 * GiB_bytes,
            non_kv_cache_memory=12 * GiB_bytes,
            cudagraph_estimate=2 * GiB_bytes,
            gpu_memory_utilization=0.9,
        )
        with self._patched(worker):
            available = worker.determine_available_memory()

        assert available == 40 * GiB_bytes - 12 * GiB_bytes
