# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ReSET temperature configuration surface.

Covers request admission: which parameter combinations are accepted, how they
are validated, and that a request enabling only the answer-phase temperature
is still tracked by the committed-marker cache.

The per-sequence policy itself is covered in `test_reset_policy.py`, and the
batched logits processor in `test_reset_logitsproc.py`.
"""

import numpy as np
import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

MAX_TEMPERATURE = 2.0


class TestMarkerCacheGate:
    """Answer-phase temperature tracking lives in SamplingStates, while
    `ThinkingBudgetState.use_thinking_budget` is what gates
    `_requires_logits_processing` for reasoning-controlled requests.
    """

    def _state(self, **kwargs):
        state = ThinkingBudgetState.__new__(ThinkingBudgetState)
        state.max_num_reqs = 4
        state.use_thinking_budget = np.zeros(4, dtype=bool)
        for name, idx in kwargs.items():
            getattr(state, name)[idx] = True
        return state

    def test_thinking_budget_enters_logits_processing(self):
        """A request carrying a thinking budget must not return early."""
        state = self._state(use_thinking_budget=0)
        assert state.use_thinking_budget[0]

    def test_untouched_request_is_not_tracked(self):
        state = self._state()
        assert not state.use_thinking_budget.any()

    def test_no_controls_tracks_nothing(self):
        assert not self._state().use_thinking_budget.any()


class TestSamplingParamsValidation:
    def test_defaults_are_unset(self):
        params = SamplingParams()
        assert params.temperature_low is None
        assert params.temperature_high is None
        assert params.entropy_threshold is None
        assert params.reset_window is None
        assert params.reasoning_answer_temperature is None
        assert not params.has_dynamic_temperature

    def test_valid_reset_schedule_is_accepted(self):
        params = SamplingParams(
            temperature=1.0,
            temperature_low=0.1,
            temperature_high=1.0,
            entropy_threshold=0.6,
            reset_window=32,
        )
        assert params.has_temperature_schedule
        assert params.has_dynamic_temperature

    @pytest.mark.parametrize("value", [0.0, -0.1, 2.1, float("nan")])
    def test_temperature_low_must_be_in_range(self, value):
        with pytest.raises(VLLMValidationError):
            SamplingParams(temperature_low=value)

    @pytest.mark.parametrize("value", [0.0, -0.1, 2.1, float("nan")])
    def test_temperature_high_must_be_in_range(self, value):
        with pytest.raises(VLLMValidationError):
            SamplingParams(temperature_high=value)

    @pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
    def test_entropy_threshold_rejects_non_positive_or_nonfinite(self, value):
        with pytest.raises(VLLMValidationError):
            SamplingParams(entropy_threshold=value)

    def test_entropy_threshold_allows_values_above_one(self):
        # tau0 is entropy in nats and is not capped at 1.
        params = SamplingParams(temperature=1.0, entropy_threshold=1.5)
        assert params.entropy_threshold == 1.5

    @pytest.mark.parametrize("steps", [0, -5, True, 2.5])
    def test_reset_window_must_be_a_positive_int(self, steps):
        with pytest.raises(VLLMValidationError):
            SamplingParams(reset_window=steps)

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
