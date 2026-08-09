# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-aware temperature scaling.

The invariant these guard hardest is that a request which sets none of the new
parameters is scaled by exactly the value it would have been scaled by before
the feature existed.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.temperature import (
    TemperatureSchedule,
    divide_by_temperature,
    resolve_temperature,
)
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

MAX_TEMPERATURE = 2.0


def make_schedule(
    base: list[float],
    final: list[float] | None = None,
    anneal_steps: list[int] | None = None,
    schedule_enabled: list[int] | None = None,
    answer_temperature: list[float] | None = None,
    answer_enabled: list[int] | None = None,
    generated_steps: list[int] | None = None,
    reasoning_phase: list[int] | None = None,
    entered_reasoning: list[int] | None = None,
) -> TemperatureSchedule:
    n = len(base)
    zeros_f = [0.0] * n
    zeros_i = [0] * n

    def f(v):
        return torch.tensor(v if v is not None else zeros_f, dtype=torch.float32)

    def i(v):
        return torch.tensor(v if v is not None else zeros_i, dtype=torch.int32)

    return TemperatureSchedule(
        base=f(base),
        final=f(final),
        anneal_steps=i(anneal_steps),
        schedule_enabled=i(schedule_enabled),
        answer_temperature=f(answer_temperature),
        answer_enabled=i(answer_enabled),
        generated_steps=i(generated_steps),
        reasoning_phase=i(reasoning_phase),
        entered_reasoning=i(entered_reasoning),
    )


class TestUnsetParamsReproduceBaseline:
    """A request that configures nothing new must be untouched."""

    def test_resolved_temperature_is_base_bit_for_bit(self):
        base = [0.0, 0.7, 1.0, 1.999, 0.01]
        schedule = make_schedule(base)
        resolved = resolve_temperature(schedule)
        assert torch.equal(resolved, schedule.base)

    def test_greedy_zero_is_not_rewritten(self):
        resolved = resolve_temperature(make_schedule([0.0, 1.0]))
        assert resolved[0].item() == 0.0

    def test_divide_matches_legacy_expression(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 16)
        temp = torch.tensor([0.0, 0.5, 1.0, 2.0])
        expected = logits / torch.where(temp < 1e-5, 1.0, temp).unsqueeze(1)
        got = divide_by_temperature(logits.clone(), temp, all_random=False)
        assert torch.equal(got, expected)

    def test_step_count_is_ignored_without_a_schedule(self):
        schedule = make_schedule([0.9] * 3, generated_steps=[0, 17, 9999])
        resolved = resolve_temperature(schedule)
        assert torch.equal(resolved, schedule.base)

    @settings(max_examples=50, deadline=None)
    @given(
        base=st.lists(
            st.floats(min_value=0.0, max_value=MAX_TEMPERATURE, width=32),
            min_size=1,
            max_size=8,
        ),
        steps=st.integers(min_value=0, max_value=10_000),
    )
    def test_unset_is_identity_for_any_base_and_step(self, base, steps):
        schedule = make_schedule(base, generated_steps=[steps] * len(base))
        assert torch.equal(resolve_temperature(schedule), schedule.base)


class TestAnneal:
    def test_interpolates_linearly(self):
        schedule = make_schedule(
            base=[1.0],
            final=[0.5],
            anneal_steps=[10],
            schedule_enabled=[1],
            generated_steps=[4],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.8)

    def test_starts_at_base(self):
        schedule = make_schedule(
            base=[1.0],
            final=[0.5],
            anneal_steps=[10],
            schedule_enabled=[1],
            generated_steps=[0],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(1.0)

    @pytest.mark.parametrize("steps", [10, 11, 250, 10_000])
    def test_clamps_at_the_anneal_length(self, steps):
        schedule = make_schedule(
            base=[1.0],
            final=[0.5],
            anneal_steps=[10],
            schedule_enabled=[1],
            generated_steps=[steps],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.5)

    def test_can_anneal_upwards(self):
        schedule = make_schedule(
            base=[0.2],
            final=[1.2],
            anneal_steps=[4],
            schedule_enabled=[1],
            generated_steps=[2],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.7)

    def test_can_reach_greedy(self):
        schedule = make_schedule(
            base=[1.0],
            final=[0.0],
            anneal_steps=[8],
            schedule_enabled=[1],
            generated_steps=[8],
        )
        assert resolve_temperature(schedule)[0].item() == 0.0

    @settings(max_examples=100, deadline=None)
    @given(
        base=st.floats(min_value=0.0, max_value=MAX_TEMPERATURE, width=32),
        final=st.floats(min_value=0.0, max_value=MAX_TEMPERATURE, width=32),
        anneal=st.integers(min_value=1, max_value=4096),
        step=st.integers(min_value=0, max_value=100_000),
    )
    def test_always_within_the_endpoints(self, base, final, anneal, step):
        schedule = make_schedule(
            base=[base],
            final=[final],
            anneal_steps=[anneal],
            schedule_enabled=[1],
            generated_steps=[step],
        )
        got = resolve_temperature(schedule)[0].item()
        assert min(base, final) - 1e-5 <= got <= max(base, final) + 1e-5
        if step >= anneal:
            assert got == pytest.approx(final, abs=1e-5)


class TestReasoningPhaseOverride:
    def test_applies_after_thinking_completes(self):
        schedule = make_schedule(
            base=[1.0],
            answer_temperature=[0.3],
            answer_enabled=[1],
            reasoning_phase=[0],
            entered_reasoning=[1],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.3)

    def test_does_not_apply_while_still_thinking(self):
        schedule = make_schedule(
            base=[1.0],
            answer_temperature=[0.3],
            answer_enabled=[1],
            reasoning_phase=[1],
            entered_reasoning=[1],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(1.0)

    def test_never_applies_if_reasoning_never_started(self):
        schedule = make_schedule(
            base=[1.0],
            answer_temperature=[0.3],
            answer_enabled=[1],
            reasoning_phase=[0],
            entered_reasoning=[0],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(1.0)

    def test_override_wins_over_the_anneal(self):
        schedule = make_schedule(
            base=[1.0],
            final=[0.5],
            anneal_steps=[10],
            schedule_enabled=[1],
            generated_steps=[10],
            answer_temperature=[0.9],
            answer_enabled=[1],
            reasoning_phase=[0],
            entered_reasoning=[1],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.9)

    def test_anneal_still_applies_during_thinking(self):
        schedule = make_schedule(
            base=[1.0],
            final=[0.5],
            anneal_steps=[10],
            schedule_enabled=[1],
            generated_steps=[10],
            answer_temperature=[0.9],
            answer_enabled=[1],
            reasoning_phase=[1],
            entered_reasoning=[1],
        )
        assert resolve_temperature(schedule)[0].item() == pytest.approx(0.5)


class TestMixedBatch:
    def test_rows_at_different_steps_resolve_independently(self):
        schedule = make_schedule(
            base=[1.0, 1.0, 0.0, 1.0, 0.8],
            final=[0.0, 0.7, 0.0, 0.5, 0.0],
            anneal_steps=[0, 10, 0, 4, 0],
            schedule_enabled=[0, 1, 0, 1, 0],
            answer_temperature=[0.0, 0.0, 0.0, 0.2, 0.3],
            answer_enabled=[0, 0, 0, 1, 1],
            generated_steps=[7, 5, 3, 100, 9],
            reasoning_phase=[0, 0, 0, 0, 1],
            entered_reasoning=[0, 0, 0, 1, 1],
        )
        got = resolve_temperature(schedule).tolist()
        assert got[0] == pytest.approx(1.0)  # unscheduled
        assert got[1] == pytest.approx(0.85)  # half way down the ramp
        assert got[2] == 0.0  # greedy, preserved
        assert got[3] == pytest.approx(0.2)  # answer override
        assert got[4] == pytest.approx(0.8)  # still thinking

    def test_scheduled_rows_do_not_disturb_unscheduled_ones(self):
        base = [0.6, 1.0, 1.3]
        plain = resolve_temperature(make_schedule(base))
        mixed = resolve_temperature(
            make_schedule(
                base=base,
                final=[0.0, 0.1, 0.0],
                anneal_steps=[0, 5, 0],
                schedule_enabled=[0, 1, 0],
                generated_steps=[3, 3, 3],
            )
        )
        assert mixed[0].item() == plain[0].item()
        assert mixed[2].item() == plain[2].item()
        assert mixed[1].item() != plain[1].item()


class TestStepOffset:
    """Speculative draft rows must each sit at their own step."""

    def test_offset_advances_the_ramp_per_row(self):
        schedule = make_schedule(
            base=[1.0] * 3,
            final=[0.0] * 3,
            anneal_steps=[10] * 3,
            schedule_enabled=[1] * 3,
            generated_steps=[0, 0, 0],
        )
        offset = torch.tensor([0, 1, 2], dtype=torch.int32)
        got = resolve_temperature(schedule, offset).tolist()
        assert got == pytest.approx([1.0, 0.9, 0.8])

    def test_offset_is_ignored_without_a_schedule(self):
        schedule = make_schedule([0.7] * 3)
        offset = torch.tensor([0, 1, 2], dtype=torch.int32)
        assert torch.equal(resolve_temperature(schedule, offset), schedule.base)


class TestMarkerCacheGate:
    """A request whose only reasoning control is the phase temperature must
    still be tracked, or the committed-marker cache never refreshes and the
    override can never fire."""

    def _state(self, **kwargs):
        state = ThinkingBudgetState.__new__(ThinkingBudgetState)
        state.max_num_reqs = 4
        for name in (
            "use_thinking_budget",
            "use_marker_penalty",
            "use_answer_reserve",
            "use_phase_temperature",
        ):
            setattr(state, name, np.zeros(4, dtype=bool))
        for name, idx in kwargs.items():
            getattr(state, name)[idx] = True
        return state

    def test_phase_temperature_only_request_is_tracked(self):
        state = self._state(use_phase_temperature=0)
        assert state.tracked_np[0]

    def test_untouched_request_is_not_tracked(self):
        state = self._state(use_phase_temperature=0)
        assert not state.tracked_np[1:].any()

    def test_no_controls_tracks_nothing(self):
        assert not self._state().tracked_np.any()


class TestSamplingParamsValidation:
    def test_defaults_are_unset(self):
        params = SamplingParams()
        assert params.temperature_final is None
        assert params.temperature_anneal_steps is None
        assert params.reasoning_answer_temperature is None
        assert not params.has_dynamic_temperature

    def test_valid_schedule_is_accepted(self):
        params = SamplingParams(
            temperature=1.0, temperature_final=0.7, temperature_anneal_steps=64
        )
        assert params.has_temperature_schedule
        assert params.has_dynamic_temperature

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature_final": 0.7},
            {"temperature_anneal_steps": 64},
        ],
    )
    def test_schedule_fields_must_come_together(self, kwargs):
        with pytest.raises(VLLMValidationError, match="must be supplied together"):
            SamplingParams(**kwargs)

    @pytest.mark.parametrize("steps", [0, -5, True, 2.5])
    def test_anneal_steps_must_be_a_positive_int(self, steps):
        with pytest.raises(VLLMValidationError):
            SamplingParams(temperature_final=0.7, temperature_anneal_steps=steps)

    @pytest.mark.parametrize("value", [-0.1, 2.1, float("nan"), float("inf")])
    def test_final_temperature_must_be_in_range(self, value):
        with pytest.raises(VLLMValidationError):
            SamplingParams(temperature_final=value, temperature_anneal_steps=8)

    def test_answer_temperature_requires_reasoning_config(self):
        with pytest.raises(VLLMValidationError, match="requires reasoning"):
            SamplingParams(reasoning_answer_temperature=0.3)

    def test_answer_temperature_with_thinking_budget_is_accepted(self):
        params = SamplingParams(
            thinking_token_budget=128, reasoning_answer_temperature=0.3
        )
        assert params.reasoning_answer_temperature == 0.3
        assert params.has_dynamic_temperature

    @pytest.mark.parametrize("value", [-0.1, 2.1, float("nan")])
    def test_answer_temperature_must_be_in_range(self, value):
        with pytest.raises(VLLMValidationError):
            SamplingParams(
                thinking_token_budget=128, reasoning_answer_temperature=value
            )
