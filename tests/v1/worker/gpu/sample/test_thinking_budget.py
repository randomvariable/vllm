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

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
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


def test_sampler_passes_add_request_args_the_holder_expects():
    """Pin the Sampler -> ThinkingBudgetState add_request contract.

    The holder takes (req_idx, prompt_len, all_token_ids, sampling_params) but
    the sampler passed only three of those, so every request raised TypeError
    before it could even reach the NameError. An AST walk inside one function
    cannot see a cross-module arity mismatch; comparing signatures can, and
    needs no GPU.
    """
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    holder_params = list(
        inspect.signature(tb.ThinkingBudgetState.add_request).parameters
    )
    sampler_params = list(inspect.signature(Sampler.add_request).parameters)
    assert holder_params == sampler_params, (
        f"Sampler.add_request{sampler_params} cannot drive "
        f"ThinkingBudgetState.add_request{holder_params}"
    )


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

    state.remove_request(0)

    # remove_request must stage a clear for every field a prior occupant could
    # have dirtied, KMP progress and force output included. Assert on the
    # staged writes rather than the flushed device values: applying them needs
    # a Triton launch, and this invariant is about what gets staged.
    for name in (
        "in_think",
        "in_end",
        "think_count",
        "countdown",
        "end_count",
        "seen_len",
        "kmp_start",
        "kmp_end",
        "force_active",
        "force_offset",
        "force_end_count",
    ):
        tensor = getattr(state, name)
        assert 0 in tensor._staged_write_indices, f"{name} not cleared"

    assert not state._has_tracked[0]
    assert state._marker_penalty_cpu[0] == 0.0


@requires_cuda
def test_kernel_written_state_is_not_uva_backed():
    """Kernel-written state must not live in a UvaBackedTensor.

    ``UvaBackedTensor.np`` is not aliased to ``.gpu``: ``copy_to_uva()`` copies
    the host array into a pooled device buffer and rebinds ``.gpu``. For state
    the kernel writes, that means (a) every host read is stale and (b) the next
    flush overwrites the kernel's accumulated device state, resetting in-flight
    requests to their prompt-scan values. Under continuous batching a flush
    happens on most steps, so budgets never reached exhaustion and forcing
    never fired.

    Asserting the type is the cheap invariant: StagedWriteTensor applies only
    the rows actually staged and leaves everything else on device untouched.
    """
    req_states = SimpleNamespace(max_num_reqs=4, device=torch.device("cuda"))
    state = tb.ThinkingBudgetState(req_states, reasoning_config=_MockReasoningConfig())

    kernel_written = (
        "in_think",
        "in_end",
        "think_count",
        "countdown",
        "end_count",
        "seen_len",
        "kmp_start",
        "kmp_end",
        "force_active",
        "force_offset",
        "force_end_count",
    )
    for name in kernel_written:
        tensor = getattr(state, name)
        assert isinstance(tensor, StagedWriteTensor), (
            f"{name} is {type(tensor).__name__}; kernel-written state in a "
            "UvaBackedTensor is clobbered on the next flush"
        )


@requires_cuda
def test_flush_without_new_requests_launches_nothing():
    """A step with no new requests must not touch device state at all.

    This is the other half of the clobber fix: ``apply_write`` early-returns
    when nothing is staged, so a decode-only step cannot disturb in-flight
    counters even before considering which rows it would write.
    """
    req_states = SimpleNamespace(max_num_reqs=4, device=torch.device("cuda"))
    state = tb.ThinkingBudgetState(req_states, reasoning_config=_MockReasoningConfig())

    for name in ("countdown", "think_count", "kmp_start"):
        assert not getattr(state, name)._staged_write_indices

    # No add_request has run, so no rows are staged and no kernel may launch.
    state.apply_staged_writes()


def test_sampler_module_has_no_unresolved_names():
    """Catch construction-time NameErrors in the MRV2 Sampler.

    9fedae27 renamed ``LogprobTokenIdsState`` to ``LogitTokenIdsState`` at the
    call site only, so constructing the sampler raised NameError for *every*
    MRV2 model, not just reasoning ones. Production escaped it purely because
    the deployed image predated the commit. Nothing else in the suite
    constructs this class, so a static resolution check is the cheap guard.
    """
    import vllm.v1.worker.gpu.sample.sampler as sampler_mod

    with open(sampler_mod.__file__) as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)

    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler"
    )
    unresolved: dict[str, list[str]] = {}
    for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
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
        missing = sorted(loaded - params - assigned - imported - set(dir(builtins)))
        if missing:
            unresolved[fn.name] = missing

    assert not unresolved, f"Sampler has unresolved names: {unresolved}"
