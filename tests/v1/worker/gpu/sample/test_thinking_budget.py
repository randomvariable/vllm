# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork extensions to the MRV2 ``ThinkingBudgetState``.

``vllm/v1/worker/gpu/sample/thinking_budget.py``

Upstream's own budget/forcing behaviour is covered by
``tests/v1/worker/test_gpu_thinking_budget.py``. This suite covers only what
this fork adds on top:

* the hesitation-marker penalty, including multi-token markers;
* the answer reserve;
* the ``Sampler._requires_logits_processing`` gate, without which a request
  that sets *only* reasoning controls never reaches the budget at all.
"""

import ast
import builtins

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import thinking_budget as tb
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
START = 90
END = 91
VOCAB_SIZE = 128

# "let me think" stands in for a real multi-token marker.
MARKER_MULTI = [70, 71, 72]
MARKER_SINGLE = [42]


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]
    reasoning_marker_token_ids = [MARKER_SINGLE]


class MockMultiTokenMarkerConfig(MockReasoningConfig):
    reasoning_marker_token_ids = [MARKER_MULTI, MARKER_SINGLE]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor(
        [3] * len(input_ids), dtype=torch.int32, device=DEVICE
    )
    state.apply(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping.cpu().numpy(),
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.tensor(local_pos, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_markers_flatten_with_offsets():
    """Markers of differing length share one flat buffer plus CSR offsets.

    The kernel indexes marker ``i`` as ``[offsets[i], offsets[i + 1])``, so a
    three-token marker followed by a one-token marker must produce offsets
    ``[0, 3, 4]`` rather than one padded slot per marker.
    """
    req_states = _make_req_states([1, START])
    state = ThinkingBudgetState(req_states, MockMultiTokenMarkerConfig())

    assert state._num_markers == 2
    assert state._marker_offsets.tolist() == [0, 3, 4]
    assert state._marker_tokens.tolist() == MARKER_MULTI + MARKER_SINGLE


def test_overlong_marker_is_dropped_not_truncated():
    """A marker that would overflow the shared token buffer is dropped.

    Truncating instead would leave the kernel matching a prefix of a marker the
    user never configured, penalising an unrelated token.
    """

    class _Overlong(MockReasoningConfig):
        reasoning_marker_token_ids = [
            list(range(100, 100 + tb._MAX_PENALTY_MARKER_TOKENS)),
            MARKER_SINGLE,
        ]

    req_states = _make_req_states([1, START])
    state = ThinkingBudgetState(req_states, _Overlong())

    assert state._num_markers == 1
    assert state._marker_offsets.tolist() == [0, tb._MAX_PENALTY_MARKER_TOKENS]


def test_multi_token_marker_penalises_only_the_completing_token():
    """The penalty lands on the final token, and only where the prefix matches.

    History ends in ``[70, 71]``, so ``72`` would complete the marker and must
    be penalised, while the prefix tokens themselves must not be -- they are
    legitimate continuations of other words. A flat per-token penalty cannot
    tell these apart; that distinction is what this pins.
    """
    req_states = _make_req_states([1, START, 70, 71])
    state = ThinkingBudgetState(req_states, MockMultiTokenMarkerConfig())
    state.add_request(3, SamplingParams(reasoning_marker_penalty=2.5))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB_SIZE, device=DEVICE)
    out = _apply(state, logits, [71], [0])

    assert out[0, 72].item() == pytest.approx(-2.5)
    assert out[0, 70].item() == pytest.approx(0.0)
    assert out[0, 71].item() == pytest.approx(0.0)
    # The single-token marker needs no prefix, so it is always penalised.
    assert out[0, 42].item() == pytest.approx(-2.5)


def test_marker_penalty_not_applied_without_matching_prefix():
    """Unrelated history leaves the marker's final token alone."""
    req_states = _make_req_states([1, START, 5, 6])
    state = ThinkingBudgetState(req_states, MockMultiTokenMarkerConfig())
    state.add_request(3, SamplingParams(reasoning_marker_penalty=2.5))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB_SIZE, device=DEVICE)
    out = _apply(state, logits, [6], [0])

    assert out[0, 72].item() == pytest.approx(0.0)
    assert out[0, 42].item() == pytest.approx(-2.5)


def test_marker_penalty_skipped_outside_think_block():
    """No penalty applies once the request has left the reasoning block."""
    req_states = _make_req_states([1, START, 10, END, 20])
    state = ThinkingBudgetState(req_states, MockMultiTokenMarkerConfig())
    state.add_request(3, SamplingParams(reasoning_marker_penalty=2.5))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB_SIZE, device=DEVICE)
    out = _apply(state, logits, [20], [0])

    assert torch.count_nonzero(out).item() == 0


def test_answer_reserve_forces_end_when_output_budget_nearly_spent():
    """The reserve forces the end marker with only the reserve left to write.

    ``max_tokens=8`` with four tokens produced and a reserve of four leaves
    exactly the reserve, so thinking must end here even though no
    ``thinking_token_budget`` was set.
    """
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(max_tokens=8, reasoning_answer_reserve=4))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB_SIZE, device=DEVICE)
    out = _apply(state, logits, [12], [0])

    assert out[0].argmax().item() == END


def test_answer_reserve_inactive_while_output_budget_remains():
    """With room to spare the reserve must not force anything."""
    req_states = _make_req_states([1, START, 10], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(max_tokens=64, reasoning_answer_reserve=4))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB_SIZE, device=DEVICE)
    out = _apply(state, logits, [10], [0])

    assert torch.count_nonzero(out).item() == 0


def test_reasoning_controls_alone_enter_the_logits_path():
    """``tracked_np`` must flag requests whose only control is a reasoning one.

    ``Sampler._requires_logits_processing`` returns early for a request with no
    logit bias, penalties, bad words, and default temperature/top-p. A greedy
    request that sets only a thinking budget matches that description, so
    without this signal the budget is silently inert -- which is the common
    case for a reasoning model.
    """
    req_states = _make_req_states([1, START])
    state = ThinkingBudgetState(req_states, MockReasoningConfig())

    assert not state.tracked_np[3]
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    assert state.tracked_np[3]

    state.add_request(2, SamplingParams(reasoning_marker_penalty=1.0))
    assert state.tracked_np[2]

    state.add_request(1, SamplingParams(max_tokens=8, reasoning_answer_reserve=4))
    assert state.tracked_np[1]

    state.add_request(0, SamplingParams())
    assert not state.tracked_np[0]


def test_sampler_gate_consults_thinking_budget():
    """Pin the gate call itself, not just the signal it reads.

    ``tracked_np`` being correct is useless if the sampler never asks. This is
    the cross-module half of the same defect, and needs no GPU state to check.
    """
    import vllm.v1.worker.gpu.sample.sampler as sampler_mod

    with open(sampler_mod.__file__) as f:
        tree = ast.parse(f.read())

    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler"
    )
    gate = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_requires_logits_processing"
    )
    reads = {n.attr for n in ast.walk(gate) if isinstance(n, ast.Attribute)}
    assert "tracked_np" in reads, (
        "_requires_logits_processing ignores the thinking budget, so a request "
        "using only reasoning controls returns before it is applied"
    )


def test_sampler_module_has_no_unresolved_names():
    """Catch construction-time NameErrors in the MRV2 Sampler.

    A prior rename touched the call site only, so constructing the sampler
    raised NameError for *every* MRV2 model, not just reasoning ones. Nothing
    else in the suite constructs this class, so a static check is the cheap
    guard.
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


def test_sampler_add_request_call_matches_state_signature():
    """Pin the Sampler -> ThinkingBudgetState add_request call.

    ``Sampler.add_request`` also feeds other per-request states, so its own
    signature is wider; what must line up is the call it makes. An arity
    mismatch there raises TypeError on every request, and no single-module
    static check can see across the two.
    """
    import inspect

    import vllm.v1.worker.gpu.sample.sampler as sampler_mod

    with open(sampler_mod.__file__) as f:
        tree = ast.parse(f.read())

    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_request"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "thinking_budget_state"
    )

    sig = inspect.signature(ThinkingBudgetState.add_request)
    bound = len(call.args) + len(call.keywords)
    expected = len(sig.parameters) - 1  # drop ``self``
    assert bound == expected, (
        f"sampler passes {bound} argument(s) to "
        f"ThinkingBudgetState.add_request{tuple(sig.parameters)}"
    )
