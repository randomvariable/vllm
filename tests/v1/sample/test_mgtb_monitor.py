# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.mgtb_monitor import (
    MGTBCalibration,
    MGTBConfig,
    MGTBRequestState,
    build_mgtb_calibration_key,
)
from vllm.v1.sample.ops.reset import reset_entropy
from vllm.v1.worker.gpu.sample.reasoning_monitor import ReasoningMonitor


def test_mgtb_recurrence_matches_calibrated_reference() -> None:
    calibration = MGTBCalibration(
        key="test",
        scores_by_bucket={
            0: tuple([0.0] * 100),
            1: tuple([0.0] * 100),
        },
    )
    config = MGTBConfig(
        calibration=calibration,
        beta_exponents=(0.5,),
        position_bucket_size=64,
    )
    state = MGTBRequestState(config=config)

    expected = 0.0
    for position in range(96):
        alarm = state.advance(
            entropy=0.2,
            logprob=-0.1,
            token_id=position,
            position=position,
            threshold=3.0,
            refractory_tokens=32,
        )
        if position in (63, 95):
            q = 1.0 / 101.0
            factor = 0.5 * q**-0.5
            expected = max(0.0, expected) + np.log(factor)
            assert state.statistic == pytest.approx(expected)
        else:
            assert not alarm

    assert state.alarms == 1
    assert state.sampled_tokens == 96
    assert state.emitted_tokens == 96
    assert state.deleted_tokens == 0


def test_mgtb_features_detect_confident_repetition_not_logprob_alone() -> None:
    config = MGTBConfig(beta_exponents=(0.5,))
    state = MGTBRequestState(config=config)

    for position in range(64):
        state.advance(
            entropy=0.01,
            logprob=-0.01,
            token_id=7,
            position=position,
            threshold=100.0,
            refractory_tokens=32,
        )

    features = state.window_features[-1]
    assert features[0] == pytest.approx(0.01)
    assert features[1] == pytest.approx(0.01)
    assert features[2] > 0.0
    assert features[3] == pytest.approx(0.0)
    assert state.window_scores[-1] > 0.0


def test_prompt_ngram_is_not_counted_as_generated_repetition() -> None:
    state = MGTBRequestState(
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        config=MGTBConfig(ngram_sizes=(6,)),
    )

    for position in range(64):
        token_id = (position % 6) + 1 if position < 6 else 100 + position
        state.advance(
            entropy=0.5,
            logprob=-0.5,
            token_id=token_id,
            position=position,
            threshold=100.0,
            refractory_tokens=32,
        )

    assert state.window_features[-1][2] == pytest.approx(0.0)


def test_calibration_loader_fails_closed_and_is_immutable(tmp_path) -> None:
    model = SimpleNamespace(
        model="model",
        revision="rev",
        tokenizer="tokenizer",
        tokenizer_revision="tok-rev",
    )
    key = build_mgtb_calibration_key(model, None, None)
    path = tmp_path / "mgtb.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": key,
                "scores_by_bucket": {"-1": [0.1, 0.2]},
            }
        ),
        encoding="utf-8",
    )

    calibration = MGTBCalibration.load(path, key)
    assert calibration is not None
    with pytest.raises(TypeError):
        cast(Any, calibration.scores_by_bucket)[-1] = (0.3,)
    assert MGTBCalibration.load(path, "wrong") is None


def test_calibration_key_includes_active_sampling_controls() -> None:
    model = SimpleNamespace(
        model="model",
        revision="rev",
        tokenizer="tokenizer",
        tokenizer_revision="tok-rev",
    )
    base = build_mgtb_calibration_key(model, None, None, SamplingParams())
    changed = build_mgtb_calibration_key(
        model, None, None, SamplingParams.from_optional(temperature=0.5, top_k=10)
    )

    assert base != changed


@pytest.mark.parametrize(
    "scores_by_bucket",
    [{}, {"-1": []}, {"-1": "not-a-list"}],
)
def test_calibration_loader_rejects_empty_or_malformed_scores(
    tmp_path, scores_by_bucket
) -> None:
    path = tmp_path / "invalid-mgtb.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "key",
                "scores_by_bucket": scores_by_bucket,
            }
        ),
        encoding="utf-8",
    )

    assert MGTBCalibration.load(path) is None


def test_reasoning_monitor_disabled_path_is_false() -> None:
    monitor = cast(Any, ReasoningMonitor.__new__(ReasoningMonitor))
    monitor._monitor_enabled = np.zeros(2, dtype=bool)
    monitor._enabled_count = 0
    monitor.enabled = SimpleNamespace(np=np.zeros(2, dtype=np.int32))
    monitor._dirty = False

    assert not monitor.has_enabled_requests
    monitor.add_request(0, SamplingParams.from_optional(reasoning_monitor=False))
    assert not monitor.monitoring(np.array([0]))[0]
    assert not monitor.has_enabled_requests

    monitor.add_request(1, SamplingParams.from_optional(reasoning_monitor=True))
    assert monitor.monitoring(np.array([1]))[0]
    assert monitor.has_enabled_requests

    monitor.add_request(1, SamplingParams.from_optional(reasoning_monitor=False))
    assert not monitor.has_enabled_requests


def test_capture_rejects_non_one_to_one_alignment() -> None:
    monitor = ReasoningMonitor.__new__(ReasoningMonitor)
    observations = monitor.capture(
        raw_logits=torch.zeros(2, 4),
        processed_logits=None,
        sampled_token_ids=torch.tensor([1]),
        committed_counts=np.array([1, 1], dtype=np.int64),
        commit_offsets=np.array([0, 0], dtype=np.int64),
    )
    assert observations == [[]]


def test_capture_spec_keeps_only_committed_prefix(monkeypatch) -> None:
    monitor = ReasoningMonitor.__new__(ReasoningMonitor)
    monitor._monitor_enabled = cast(Any, np.array([True], dtype=bool))

    def fake_compute_token_logprobs(logits, token_ids):
        return torch.log_softmax(logits, dim=-1).gather(1, token_ids)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs",
        fake_compute_token_logprobs,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.reasoning_monitor.reset_entropy",
        lambda logits: torch.full((logits.shape[0],), 0.5, device=logits.device),
    )

    observations = monitor.capture_spec(
        raw_logits=torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]
        ),
        control_logits=torch.tensor(
            [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
        ),
        sampled_token_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        committed_counts=np.array([1], dtype=np.int64),
        positions=torch.tensor([10, 11, 12], dtype=torch.int32),
        cu_num_logits=np.array([0, 3], dtype=np.int64),
        request_indices=np.array([0], dtype=np.int64),
    )

    assert len(observations) == 1
    assert len(observations[0]) == 1
    observation = observations[0][0]
    assert observation.token_id == 0
    assert observation.position == 10


def test_capture_records_plain_decode_observation(monkeypatch) -> None:
    monitor = ReasoningMonitor.__new__(ReasoningMonitor)
    monitor._monitor_enabled = cast(Any, np.array([True], dtype=bool))

    def fake_compute_token_logprobs(logits, token_ids):
        return torch.log_softmax(logits, dim=-1).gather(1, token_ids)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs",
        fake_compute_token_logprobs,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.reasoning_monitor.reset_entropy",
        lambda logits: torch.full((logits.shape[0],), 0.5, device=logits.device),
    )

    observations = monitor.capture(
        raw_logits=torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
        processed_logits=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        sampled_token_ids=torch.tensor([1], dtype=torch.long),
        committed_counts=np.array([1], dtype=np.int64),
        commit_offsets=np.array([7], dtype=np.int64),
        request_indices=np.array([0], dtype=np.int64),
    )

    assert len(observations[0]) == 1
    assert observations[0][0].token_id == 1
    assert observations[0][0].position == 7


def test_reset_entropy_upcasts_reduced_precision() -> None:
    entropy = reset_entropy(torch.zeros(2, 8, dtype=torch.bfloat16))
    assert entropy.dtype is torch.float32
    assert torch.allclose(entropy, torch.full((2,), math.log(8.0)), atol=1e-3)


def test_capture_spec_accepts_bfloat16_logits(monkeypatch) -> None:
    """Model logits are bfloat16 on the serving path.

    ``numpy`` has no bfloat16 dtype, so an un-upcast observable raises
    ``TypeError: Got unsupported ScalarType BFloat16`` inside the sampler and
    kills EngineCore for every request that enables the monitor. ``reset_entropy``
    stays real here because it is the primitive that produced the crash; only the
    Triton logprob kernel is stubbed, and it returns bfloat16 so the conversion
    guard on the chosen-token logprob is exercised too.
    """
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs",
        lambda logits, token_ids: torch.log_softmax(logits.float(), dim=-1)
        .gather(1, token_ids)
        .to(torch.bfloat16),
    )
    monitor = ReasoningMonitor.__new__(ReasoningMonitor)
    monitor._monitor_enabled = cast(Any, np.array([True], dtype=bool))

    observations = monitor.capture_spec(
        raw_logits=torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        control_logits=torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        sampled_token_ids=torch.tensor([[0]], dtype=torch.long),
        committed_counts=np.array([1], dtype=np.int64),
        positions=torch.tensor([7], dtype=torch.int32),
        cu_num_logits=np.array([0, 1], dtype=np.int64),
        request_indices=np.array([0], dtype=np.int64),
    )

    observation = observations[0][0]
    assert math.isfinite(observation.entropy_pre)
    assert observation.entropy_pre > 0.0
    assert math.isfinite(observation.entropy_post)
    assert math.isfinite(observation.logprob)
    assert observation.logprob < 0.0
    assert observation.position == 7
