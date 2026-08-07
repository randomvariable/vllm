# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MRV2 ``ThinkingBudgetState``.

``vllm/v1/worker/gpu/sample/thinking_budget.py``

Phase 0 regression coverage:

* ``apply_thinking_budget`` used to pass ``cu_num_logits`` — a name that was
  never a parameter, never assigned in the body, and never a module global.
  The call is unconditional past the tracking guard, so **any** request with a
  budget/reserve/marker raised ``NameError``. The state was only never seen
  because every production request served through the MRV1 holder instead.
  These tests pin both the static guarantee and the call-site fix.
* ``remove_request`` used to leave ``kmp_start``, ``kmp_end``, ``force_offset``
  and ``force_end_count`` from a previous occupant of a recycled slot.
"""

import ast
import builtins
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.sample import thinking_budget as tb

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class _MockReasoningConfig:
    """Minimal reasoning config for constructing an enabled state."""

    enabled = True
    reasoning_start_token_ids = [10]
    reasoning_end_token_ids = [11]
    reasoning_marker_token_ids = [42]


def _make_req_states(max_num_reqs: int = 4) -> SimpleNamespace:
    return SimpleNamespace(max_num_reqs=max_num_reqs, device=torch.device("cpu"))


def test_apply_thinking_budget_has_no_unresolved_names():
    """Guard the exact defect class that shipped the NameError.

    AST-walks ``apply_thinking_budget`` and asserts every loaded name is
    either a parameter, locally assigned, a module global, or a builtin.
    This catches an undefined ``cu_num_logits`` (or any future equivalent)
    without needing a GPU.
    """
    src = inspect.getsource(tb.ThinkingBudgetState.apply_thinking_budget)
    fn = ast.parse(src.lstrip()).body[0]
    assert isinstance(fn, ast.FunctionDef)
    params = {a.arg for a in fn.args.args}
    loaded = {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assigned = {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    unresolved = loaded - params - assigned - set(vars(tb)) - set(dir(builtins))
    assert unresolved == set(), f"unresolved names: {sorted(unresolved)}"


def test_apply_forcing_signature_has_no_cu_num_logits():
    sig = inspect.signature(tb.ThinkingBudgetState._apply_forcing)
    assert "cu_num_logits" not in sig.parameters


def test_apply_forcing_body_does_not_reference_cu_num_logits():
    src = inspect.getsource(tb.ThinkingBudgetState)
    assert "cu_num_logits" not in src


def test_disabled_state_returns_logits_unchanged():
    # ``reasoning_config=None`` -> not enabled -> early return before any
    # buffer construction, so this runs on a CPU-only box.
    req_states = _make_req_states()
    state = tb.ThinkingBudgetState(req_states, reasoning_config=None)
    assert not state._enabled

    logits = torch.zeros(2, 8)
    out = state.apply_thinking_budget(logits, None, None, None, None)
    assert out is logits  # returned untouched, no kernel launch


def test_apply_with_no_tracked_requests_returns_logits_unchanged():
    # Construction is fine on CPU because ``add_request`` never marks the
    # request tracked (no budget/reserve/marker), and the method short-
    # circuits on ``has_tracked_requests`` before the Triton launch.
    req_states = _make_req_states()
    state = tb.ThinkingBudgetState(req_states, reasoning_config=None)
    assert not state.has_tracked_requests

    logits = torch.zeros(3, 8)
    out = state.apply_thinking_budget(logits, None, None, None, None)
    assert out is logits


@requires_cuda
def test_remove_request_resets_kmp_and_force_state():
    """A recycled ``req_idx`` must not inherit mid-marker progress."""
    req_states = SimpleNamespace(max_num_reqs=4, device=torch.device("cuda"))
    state = tb.ThinkingBudgetState(req_states, reasoning_config=_MockReasoningConfig())
    assert state._enabled

    # Populate then remove a tracked request.
    sp = SimpleNamespace(
        thinking_token_budget=10,
        reasoning_answer_reserve=None,
        reasoning_marker_penalty=None,
        max_tokens=100,
    )
    state.add_request(0, prompt_len=3, all_token_ids=[10, 5, 5], sampling_params=sp)
    assert state._has_tracked[0]

    # Simulate mid-marker KMP/force state leaking on the slot.
    state.kmp_start.np[0] = 1
    state.kmp_end.np[0] = 1
    state.force_offset.np[0] = 3
    state.force_end_count.np[0] = 2

    state.remove_request(0)

    assert not state._has_tracked[0]
    assert state.kmp_start.np[0] == 0
    assert state.kmp_end.np[0] == 0
    assert state.force_offset.np[0] == 0
    assert state.force_end_count.np[0] == 0
    assert state.in_think.np[0] == False  # noqa: E712
    assert state.force_active.np[0] == False  # noqa: E712
    # Config lives in StagedWriteTensors (no ``.np`` view); the CPU mirror is
    # what the host gating path reads, so assert the reset landed there.
    assert state._marker_penalty_cpu[0] == 0.0
    assert not state._has_tracked[0]
