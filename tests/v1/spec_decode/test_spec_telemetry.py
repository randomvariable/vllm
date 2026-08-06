# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from vllm.v1.spec_decode.dynamic.telemetry import (
    AcceptanceEstimator,
    ForwardCostSample,
    SpecDecodeTelemetry,
)
from vllm.v1.spec_decode.timing import SpecForwardTimings

MAX_SPEC_TOKENS = 8


@st.composite
def draft_events(draw, max_spec_tokens: int = MAX_SPEC_TOKENS):
    """Draw a (num_draft_tokens, num_accepted) pair with num_accepted <= K."""
    num_draft_tokens = draw(st.integers(min_value=1, max_value=max_spec_tokens))
    num_accepted = draw(st.integers(min_value=0, max_value=num_draft_tokens))
    return num_draft_tokens, num_accepted


event_lists = st.lists(draft_events(), min_size=0, max_size=64)


def _estimator(half_life: float = 8.0) -> AcceptanceEstimator:
    return AcceptanceEstimator(
        half_life_drafts=half_life, max_spec_tokens=MAX_SPEC_TOKENS
    )


@given(events=event_lists)
def test_acceptance_rate_within_unit_interval(events):
    estimator = _estimator()
    for num_draft_tokens, num_accepted in events:
        estimator.observe(num_draft_tokens, num_accepted)
        assert 0.0 <= estimator.acceptance_rate <= 1.0
    assert 0.0 <= estimator.acceptance_rate <= 1.0


@given(events=event_lists, extra=st.integers(min_value=1, max_value=MAX_SPEC_TOKENS))
def test_all_accepted_observation_never_decreases_rate(events, extra):
    estimator = _estimator()
    for num_draft_tokens, num_accepted in events:
        estimator.observe(num_draft_tokens, num_accepted)
    before = estimator.acceptance_rate
    estimator.observe(extra, extra)
    assert estimator.acceptance_rate >= before - 1e-12


@given(
    half_life=st.integers(min_value=2, max_value=64),
    burn_in=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=50)
def test_half_life_moves_estimate_at_least_halfway(half_life, burn_in):
    """After `half_life_drafts` contradicting drafts, move at least halfway.

    This is the guard against a per-step vs per-draft decay mixup, which
    would otherwise ship silently.
    """
    estimator = AcceptanceEstimator(
        half_life_drafts=float(half_life), max_spec_tokens=MAX_SPEC_TOKENS
    )
    for _ in range(burn_in):
        estimator.observe(MAX_SPEC_TOKENS, MAX_SPEC_TOKENS)
    start = estimator.acceptance_rate
    assume(start > 0.5)

    for _ in range(half_life):
        estimator.observe(MAX_SPEC_TOKENS, 0)

    target = 0.0
    midpoint = start + (target - start) / 2.0
    assert estimator.acceptance_rate <= midpoint + 1e-9


@given(
    events_a=st.lists(draft_events(), min_size=1, max_size=32),
    events_b=st.lists(draft_events(), min_size=1, max_size=32),
    batch_a=st.integers(min_value=1, max_value=64),
    gap=st.integers(min_value=1, max_value=64),
)
def test_bucket_isolation(events_a, events_b, batch_a, gap):
    batch_b = batch_a + gap
    telemetry = SpecDecodeTelemetry(
        max_spec_tokens=MAX_SPEC_TOKENS, half_life_drafts=8.0, max_batch_size=256
    )
    for num_draft_tokens, num_accepted in events_b:
        telemetry.observe_acceptance(batch_b, num_draft_tokens, num_accepted)
    expected = telemetry.snapshot(batch_b)

    for num_draft_tokens, num_accepted in events_a:
        telemetry.observe_acceptance(batch_a, num_draft_tokens, num_accepted)

    assert telemetry.snapshot(batch_b) == expected


@given(events=st.lists(draft_events(), min_size=1, max_size=32))
def test_effective_n_starts_at_zero_and_increases(events):
    estimator = _estimator()
    assert estimator.effective_n == 0.0
    previous = estimator.effective_n
    for num_draft_tokens, num_accepted in events:
        estimator.observe(num_draft_tokens, num_accepted)
        assert estimator.effective_n > previous
        previous = estimator.effective_n


def test_rate_at_pos_none_for_never_drafted_position():
    estimator = _estimator()
    estimator.observe(num_draft_tokens=2, num_accepted=2)

    assert estimator.rate_at_pos(0) == pytest.approx(1.0)
    assert estimator.rate_at_pos(1) == pytest.approx(1.0)
    for j in range(2, MAX_SPEC_TOKENS):
        assert estimator.rate_at_pos(j) is None


def test_rate_at_pos_out_of_range_is_none():
    estimator = _estimator()
    assert estimator.rate_at_pos(-1) is None
    assert estimator.rate_at_pos(MAX_SPEC_TOKENS) is None


def test_rate_at_pos_zero_accept_is_zero_not_none():
    estimator = _estimator()
    estimator.observe(num_draft_tokens=3, num_accepted=0)
    assert estimator.rate_at_pos(0) == pytest.approx(0.0)
    assert estimator.rate_at_pos(3) is None


def test_observe_rejects_impossible_counts():
    estimator = _estimator()
    with pytest.raises(ValueError):
        estimator.observe(num_draft_tokens=2, num_accepted=3)
    with pytest.raises(ValueError):
        estimator.observe(num_draft_tokens=MAX_SPEC_TOKENS + 1, num_accepted=0)


def test_zero_draft_tokens_is_a_no_op():
    estimator = _estimator()
    estimator.observe(num_draft_tokens=0, num_accepted=0)
    assert estimator.effective_n == 0.0
    assert estimator.acceptance_rate == 0.0


def test_cold_snapshot_reports_no_evidence():
    telemetry = SpecDecodeTelemetry(max_spec_tokens=MAX_SPEC_TOKENS)
    signals = telemetry.snapshot(4)
    assert signals.effective_n == 0.0
    assert signals.acceptance_rate == 0.0
    assert signals.mean_accept_len == 1.0
    assert signals.acceptance_per_pos == (None,) * MAX_SPEC_TOKENS
    assert signals.target_ms is None
    assert signals.steps_per_second is None


def test_forward_cost_sample_from_timings_preserves_forward_token_count():
    timings = SpecForwardTimings(
        target_ms=8.0, draft_ms=2.0, num_tokens=96, num_reqs=32, num_spec_tokens=2
    )
    sample = ForwardCostSample.from_timings(timings)
    assert sample.num_tokens == 96
    assert sample.num_reqs == 32
    assert sample.total_ms == pytest.approx(10.0)
    assert sample.steps_per_second == pytest.approx(100.0)


def test_cost_points_are_build_sps_table_shaped():
    telemetry = SpecDecodeTelemetry(max_spec_tokens=MAX_SPEC_TOKENS)
    telemetry.observe_forward(
        ForwardCostSample(num_tokens=64, num_reqs=32, target_ms=8.0, draft_ms=2.0)
    )
    telemetry.observe_forward(
        ForwardCostSample(num_tokens=128, num_reqs=64, target_ms=16.0, draft_ms=4.0)
    )
    token_points, sps_points = telemetry.cost_points()
    assert len(token_points) == len(sps_points) == 2
    assert set(token_points) == {64, 128}
    assert all(sps > 0.0 for sps in sps_points)


def test_missing_drafter_forward_leaves_draft_ms_none():
    telemetry = SpecDecodeTelemetry(max_spec_tokens=MAX_SPEC_TOKENS)
    telemetry.observe_forward(
        ForwardCostSample(num_tokens=64, num_reqs=8, target_ms=8.0, draft_ms=None)
    )
    signals = telemetry.snapshot(8)
    assert signals.target_ms == pytest.approx(8.0)
    assert signals.draft_ms is None
    assert signals.steps_per_second == pytest.approx(125.0)


def test_batch_sizes_above_max_share_the_top_bucket():
    telemetry = SpecDecodeTelemetry(
        max_spec_tokens=MAX_SPEC_TOKENS, half_life_drafts=8.0, max_batch_size=16
    )
    telemetry.observe_acceptance(1024, num_draft_tokens=4, num_accepted=4)
    assert telemetry.snapshot(16).effective_n > 0.0
