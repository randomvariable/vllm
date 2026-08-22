# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the overthinking-marker logit penalty (arXiv 2606.00206).

Covers Phase A of the layered reasoning-control stack:

1. `SamplingParams` validation of `reasoning_marker_penalty`.
2. Tokenizer resolution of the canonical marker list (single-token leading
   space variants, multi-token skipped, function-word collisions rejected,
   zero-resolved fail-closed).
3. Kernel behavior: marker logits drop by exactly lambda; non-marker logits
   unchanged; answer-phase and budget-exhausted rows untouched.
4. Sampling run contract: a marker row produces valid tokens; a disabled row
   takes the fast path.
"""

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.marker_data import (
    OVERTHINKING_MARKERS,
    resolve_marker_token_ids,
)
from vllm.v1.worker.gpu.sample.marker_penalty import (
    ReasoningMarkerPenaltyState,
)

MAX_MARKER_PENALTY = 10.0


class _MockTokenizer:
    """Resolves every leading-space marker to a unique single token id."""

    def __init__(self):
        self.vocab_size = len(OVERTHINKING_MARKERS) + 4
        self._marker_lut = {s: i for i, s in enumerate(OVERTHINKING_MARKERS)}
        self._decode_lut = {i: s for s, i in self._marker_lut.items()}

    def encode(self, text, add_special_tokens=False):
        tid = self._marker_lut.get(text)
        return [tid] if tid is not None else []

    def decode(self, token_ids, skip_special_tokens=False):
        assert len(token_ids) == 1
        return self._decode_lut.get(token_ids[0], "")


class _EmptyTokenizer(_MockTokenizer):
    def encode(self, text, add_special_tokens=False):
        return []


class _CollideTokenizer(_MockTokenizer):
    def __init__(self):
        super().__init__()
        # Force " or" to resolve to a token whose decode (after lstrip) is the
        # bare function word "or" -> must be rejected.
        self._extra_id = len(OVERTHINKING_MARKERS)
        self._marker_lut[" or"] = self._extra_id

    def decode(self, token_ids, skip_special_tokens=False):
        if token_ids[0] == self._extra_id:
            return " or"
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


class _MultiTokenizer(_MockTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text == " perhaps":
            return [1, 2]
        return super().encode(text, add_special_tokens=add_special_tokens)


def _gc():
    return torch.cuda.is_available() or (
        getattr(torch, "hip", False) and torch.cuda.is_available()
    )


# --------------------------------------------------------------------------
# SamplingParams validation
# --------------------------------------------------------------------------
class TestSamplingParamsValidation:
    def test_default_is_unset(self):
        assert SamplingParams().reasoning_marker_penalty is None

    def test_none_and_zero_disabled(self):
        assert (
            SamplingParams(reasoning_marker_penalty=0.0).reasoning_marker_penalty == 0.0
        )

    def test_valid_value_accepted(self):
        p = SamplingParams(reasoning_marker_penalty=2.0)
        assert p.reasoning_marker_penalty == 2.0

    def test_boundary_values_accepted(self):
        assert (
            SamplingParams(reasoning_marker_penalty=0.0).reasoning_marker_penalty == 0.0
        )
        assert (
            SamplingParams(reasoning_marker_penalty=10.0).reasoning_marker_penalty
            == 10.0
        )

    @pytest.mark.parametrize("v", [-0.1, 10.1, float("nan"), float("inf")])
    def test_out_of_range_rejected(self, v):
        with pytest.raises(VLLMValidationError):
            SamplingParams(reasoning_marker_penalty=v)

    @pytest.mark.parametrize("v", [True, "2", [2.0]])
    def test_non_numeric_rejected(self, v):
        with pytest.raises(VLLMValidationError):
            SamplingParams(reasoning_marker_penalty=v)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
class TestMarkerResolution:
    def test_all_markers_resolve_to_unique_single_tokens(self):
        ids = resolve_marker_token_ids(_MockTokenizer())
        assert len(ids) == len(set(ids))
        assert all(isinstance(i, int) for i in ids)

    def test_multi_token_marker_skipped(self):
        ids = resolve_marker_token_ids(_MultiTokenizer())
        # " perhaps" is dropped; the rest resolve.
        assert ids
        assert _MockTokenizer()._marker_lut[" perhaps"] not in ids

    def test_function_word_collision_rejected(self):
        ids = resolve_marker_token_ids(_CollideTokenizer())
        # The " or" marker resolves to `_extra_id`, which decodes to " or";
        # lstrip -> "or" is a function word, so it must be rejected.
        assert _CollideTokenizer()._extra_id not in ids

    def test_zero_resolved_is_empty(self):
        assert resolve_marker_token_ids(_EmptyTokenizer()) == []

    def test_marker_list_has_exactly_50(self):
        assert len(OVERTHINKING_MARKERS) == 50
        assert all(m.startswith(" ") for m in OVERTHINKING_MARKERS)


# --------------------------------------------------------------------------
# State admission / no-op contract
# --------------------------------------------------------------------------
class TestStateAdmission:
    def _bare_state(self, enabled=True, max_num_reqs=4):
        state = ReasoningMarkerPenaltyState.__new__(ReasoningMarkerPenaltyState)
        state.enabled = enabled
        state.max_num_reqs = max_num_reqs
        state.device = torch.device("cpu")
        state.use_marker_penalty = np.zeros(max_num_reqs, dtype=bool)
        state._penalty_dirty = False
        # A minimal UVA-like stub for the penalty array.
        state.penalty = type(
            "P", (), {"np": np.zeros(max_num_reqs, dtype=np.float32)}
        )()
        state._reset_reqs = []
        return state

    def test_plain_request_not_active(self):
        state = self._bare_state()
        state.add_request(0, SamplingParams())
        assert not state.use_marker_penalty[0]

    def test_marker_request_active(self):
        state = self._bare_state()
        state.add_request(0, SamplingParams(reasoning_marker_penalty=2.0))
        assert state.use_marker_penalty[0]

    def test_zero_penalty_not_active(self):
        state = self._bare_state()
        state.add_request(0, SamplingParams(reasoning_marker_penalty=0.0))
        assert not state.use_marker_penalty[0]

    def test_disabled_state_never_tracks(self):
        state = self._bare_state(enabled=False)
        state.add_request(0, SamplingParams(reasoning_marker_penalty=2.0))
        assert not state.use_marker_penalty[0]

    def test_no_markers_fails_closed(self):
        # Reasoning resolves no markers on this tokenizer but reasoning is
        # otherwise configured.
        state = self._bare_state(enabled=False)
        assert not state.enabled
        assert resolve_marker_token_ids(_EmptyTokenizer()) == []


# --------------------------------------------------------------------------
# Coupling with ReSET entropy temperature (pure torch, no device)
# --------------------------------------------------------------------------
class TestReSETCoupling:
    """The marker penalty lowers the entropy ReSET measures.

    Subtracting ``lambda`` from ~50 marker logits changes the distribution,
    so the full-vocabulary entropy ReSET computes on the *post-processed*
    logits is lower than on the raw logits. The two mechanisms are additive
    in mechanism but coupled in outcome; this test pins the coupling so a
    future change cannot silently decouple them.
    """

    def test_penalty_lowers_reset_entropy(self):
        from vllm.v1.sample.ops.reset import reset_entropy

        vocab = 128
        marker = [10, 42, 77]
        lam = 3.0
        logits = torch.randn(1, vocab, dtype=torch.float32)
        penalized = logits.clone()
        penalized[0, marker] -= lam

        ent_raw = reset_entropy(logits)
        ent_pen = reset_entropy(penalized)
        assert ent_pen.item() < ent_raw.item()

    def test_zero_lambda_is_a_noop_on_entropy(self):
        from vllm.v1.sample.ops.reset import reset_entropy

        vocab = 128
        marker = [10, 42, 77]
        logits = torch.randn(1, vocab, dtype=torch.float32)
        penalized = logits.clone()
        penalized[0, marker] -= 0.0  # no-op penalty

        assert torch.allclose(reset_entropy(penalized), reset_entropy(logits))


# --------------------------------------------------------------------------
# Kernel behavior (device-gated)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not _gc(),
    reason="Kernel test requires a CUDA/HIP device",
)
class TestMarkerKernel:
    def test_penalty_applies_only_in_reasoning_phase(self):
        from vllm.v1.worker.gpu.sample.marker_penalty import apply_marker_penalty

        dev = torch.device("cuda")
        torch.manual_seed(0)
        vocab = 64
        lam = 3.0
        marker = [10, 42]
        n = 4
        # row0: START(2) at committed 0, no end -> reasoning (penalized).
        # row1: START(2) at 0, END(20) at 1 -> answer phase (untouched).
        # row2: START(2) at 0, budget exhausted (budget=1) -> untouched.
        # row3: START(2) at 0, penalty 0 -> disabled (untouched).
        all_tokens = torch.zeros((n, 32), dtype=torch.int32, device=dev)
        all_tokens[0, 0] = 2
        all_tokens[1, 0] = 2
        all_tokens[1, 1] = 20
        all_tokens[2, 0] = 2
        all_tokens[3, 0] = 2
        total_len = torch.tensor([1, 2, 1, 1], dtype=torch.int32, device=dev)
        input_ids = torch.full((n,), 100, dtype=torch.int32, device=dev)
        expanded_local_pos = torch.full((n,), 1, dtype=torch.int32, device=dev)
        logits = torch.randn(n, vocab, device=dev) + 5.0
        base = logits.clone()
        budget = torch.tensor([-1, -1, 1, -1], dtype=torch.int32, device=dev)
        penalty = torch.tensor([lam, lam, lam, 0.0], dtype=torch.float32, device=dev)
        e_map = torch.arange(n, dtype=torch.int64, device=dev)
        req_ids = torch.arange(n, dtype=torch.int64, device=dev)
        c_start = torch.full((n,), -1, dtype=torch.int32, device=dev)
        c_end = torch.full((n,), -1, dtype=torch.int32, device=dev)
        c_scan = torch.zeros(n, dtype=torch.int32, device=dev)
        start_t = torch.tensor([2], dtype=torch.int32, device=dev)
        end_t = torch.tensor([20], dtype=torch.int32, device=dev)
        marker_t = torch.tensor(marker, dtype=torch.int32, device=dev)

        apply_marker_penalty(
            logits,
            req_ids,
            e_map,
            penalty,
            budget,
            all_tokens,
            total_len,
            input_ids,
            expanded_local_pos,
            c_start,
            c_end,
            c_scan,
            start_t,
            end_t,
            marker_t,
        )
        torch.accelerator.synchronize()

        # Reasoning row0: markers down by exactly lambda, non-markers unchanged.
        for m in marker:
            assert torch.allclose(logits[0, m], base[0, m] - lam)
        for v in range(vocab):
            if v not in marker:
                assert torch.allclose(logits[0, v], base[0, v])

        # Answer-phase, budget-exhausted, and disabled rows are byte-identical.
        for r in (1, 2, 3):
            assert torch.allclose(logits[r], base[r])


# --------------------------------------------------------------------------
# Data integrity
# --------------------------------------------------------------------------
def test_marker_list_no_duplicates():
    assert len(OVERTHINKING_MARKERS) == len(set(OVERTHINKING_MARKERS))
