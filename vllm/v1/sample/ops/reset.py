# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Faithful batched implementation of the ReSET temperature policy.

Paper: "ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware
Temperature Scaling" (arXiv 2606.13233). Reference implementation:
``github.com/aiha-lab/ReSET`` (``reset/reset/logits_processor.py``), a vLLM v1
logits processor. This module reproduces that algorithm exactly, but batched
across the whole decode batch with plain tensor ops so no per-row host
synchronisation (``.item()``) is needed and it runs unchanged on CPU or GPU.

Per decoded token, for each request with token-distribution entropy ``H_t``:

    T_t = T_low   if  H_t <  tau_t          tau_t = tau_0    if  H_step <= H_bar
    T_t = T_high  if  H_t >= tau_t          tau_t = H_step   if  H_step >  H_bar

* ``H_t`` is the Shannon entropy (nats) of ``softmax(logits)`` at temperature
  1.0 -- the *raw* logits, never a previously resolved temperature. There is
  no temperature feedback and no base temperature.
* ``H_bar`` is the running mean of the token entropies seen *before* this
  token (paper Eq. 2).
* ``H_step`` is the within-step entropy estimate (paper Eq. 3): the mean of a
  size-``w`` sliding window (which spans steps) for the first ``w`` tokens of
  a step, then the within-step running mean.
* A reasoning step ends on ``"\\n\\n"``: a boundary is hit when the last output
  token decodes to text containing a double newline, or the last two output
  tokens are each a single newline. The within-step buffer resets at a
  boundary; the sliding window and ``H_bar`` do not.

Paper vs reference: Eq. 2 defines ``H_bar`` over ``i = 0..t`` and Eq. 3 sums
the within-step average to ``t`` inclusive, both *including* the current
token's entropy. The reference implementation instead evaluates both from
state carried in and appends the current entropy afterwards, so its averages
lag the equations by one token. This module reproduces the reference, because
that is what the released results were produced with.

``T_high`` is 1.0 and ``w`` is 32 throughout the paper. ``T_low`` and ``tau_0``
(the 80th-percentile token entropy) are calibrated per (model, task); the
``T_LOW``/``TAU0`` constants below are the reference implementation's fallbacks
rather than calibrated values for any particular model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

# Reference defaults (arXiv 2606.13233; github.com/aiha-lab/ReSET).
T_HIGH = 1.0
T_LOW = 0.1
TAU0 = 0.6349
W = 32

_PROB_EPS = 1e-10


def reset_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of ``softmax(logits)`` for each row.

    Mirrors the reference ``entropy_of``: the softmax is formed from the raw
    logits (temperature 1.0) and the probabilities are floored at ``1e-10``
    inside the logarithm only.

    Args:
        logits: ``[num_rows, vocab]`` logits, taken over the full vocabulary.

    Returns:
        A ``[num_rows]`` float tensor of per-row entropies.
    """
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(_PROB_EPS).log()).sum(dim=-1)


@dataclass(eq=False)
class ResetState:
    """Row-aligned ReSET configuration and running state.

    Configuration tensors are immutable per request; the running-state tensors
    are advanced in place by `resolve_reset` every decoded token, so a view
    taken with `narrow` writes straight back into the owning allocation.

    Attributes:
        enabled: Non-zero where the ReSET policy applies to the row.
        base: Static temperature used where ReSET is not enabled.
        t_low: ``T_low`` -- temperature for a below-threshold (confident) token.
        t_high: ``T_high`` -- temperature for an at/above-threshold token.
        tau0: ``tau_0`` -- global token-entropy threshold for confident steps.
        window: ``w`` -- sliding-window / step-init length. Must be
            ``<= sw_ring.shape[-1]``.
        global_sum: Running sum of every token entropy (``H_bar`` numerator).
        global_n: Running count of tokens (``H_bar`` denominator).
        sw_ring: ``[num_rows, W]`` sliding-window ring of recent entropies.
        sw_pos: Write cursor into ``sw_ring``.
        sw_count: Number of valid sliding-window entries, capped at ``window``.
        step_sum: Within-step entropy sum since the last step boundary.
        step_len: Number of tokens accumulated into ``step_sum``.
        prev_was_nl: Non-zero if the previous output token was a single
            newline, used to detect a two-newline step boundary.
    """

    enabled: torch.Tensor
    base: torch.Tensor
    t_low: torch.Tensor
    t_high: torch.Tensor
    tau0: torch.Tensor
    window: torch.Tensor
    global_sum: torch.Tensor
    global_n: torch.Tensor
    sw_ring: torch.Tensor
    sw_pos: torch.Tensor
    sw_count: torch.Tensor
    step_sum: torch.Tensor
    step_len: torch.Tensor
    prev_was_nl: torch.Tensor

    def _names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    def clone_running(self) -> dict[str, torch.Tensor]:
        """Snapshot the running-state fields (a copy) for deferred commit."""
        return {name: getattr(self, name).clone() for name in _RUNNING_FIELDS}

    def narrow(self, num_rows: int) -> ResetState:
        """View of the first ``num_rows`` rows; advances write back in place."""
        return ResetState(**{n: getattr(self, n)[:num_rows] for n in self._names()})

    def index_select(self, rows: torch.Tensor) -> ResetState:
        """Gather one row per entry of ``rows`` (a read-only copy).

        Advanced indexing copies, so state advanced through the result is not
        written back. Use `scatter_into` to persist it.
        """
        return ResetState(**{n: getattr(self, n)[rows] for n in self._names()})

    def scatter_into(self, owner: ResetState, rows: torch.Tensor) -> None:
        """Write this view's running state back into ``owner`` at ``rows``."""
        for name in _RUNNING_FIELDS:
            getattr(owner, name).index_copy_(0, rows, getattr(self, name))


# Running-state fields advanced per committed token; config fields are
# immutable per request and never snapshotted or scattered.
_RUNNING_FIELDS = (
    "global_sum",
    "global_n",
    "sw_ring",
    "sw_pos",
    "sw_count",
    "step_sum",
    "step_len",
    "prev_was_nl",
)


def resolve_reset(
    logits: torch.Tensor,
    last_token: torch.Tensor,
    gen_step: torch.Tensor,
    nl_lut: torch.Tensor,
    dnl_lut: torch.Tensor,
    state: ResetState,
) -> torch.Tensor:
    """Resolve the ReSET temperature for every row, advancing state in place.

    Args:
        logits: ``[num_rows, vocab]`` logits about to be sampled. Entropy is
            taken over the full vocabulary at temperature 1.0.
        last_token: ``[num_rows]`` id of each row's most recent *generated*
            token. Ignored where ``gen_step == 0``.
        gen_step: ``[num_rows]`` count of tokens generated so far (``t``).
        nl_lut: ``[vocab]`` bool, true for token ids decoding to a single
            newline.
        dnl_lut: ``[vocab]`` bool, true for token ids whose text contains a
            double newline.
        state: Row-aligned `ResetState`; running fields are advanced in place.

    Returns:
        A ``[num_rows]`` float tensor of per-row temperatures. Rows whose
        request has no ReSET policy return ``state.base`` unchanged.
    """
    enabled = state.enabled != 0
    ent = reset_entropy(logits).to(state.base.dtype)

    # Step-boundary detection over *generated* tokens only.
    has_last = gen_step >= 1
    last_is_nl = nl_lut[last_token] & has_last
    last_is_dnl = dnl_lut[last_token] & has_last
    prev_nl = state.prev_was_nl != 0
    boundary = last_is_dnl | (last_is_nl & prev_nl & (gen_step >= 2))

    # A boundary resets the within-step buffer (HSE restart at t_0).
    reset_mask = enabled & boundary
    zero_sum = torch.zeros_like(state.step_sum)
    zero_len = torch.zeros_like(state.step_len)
    state.step_sum.copy_(torch.where(reset_mask, zero_sum, state.step_sum))
    state.step_len.copy_(torch.where(reset_mask, zero_len, state.step_len))

    # H_step estimate, computed before this token's entropy is folded in.
    col = torch.arange(state.sw_ring.shape[-1], device=logits.device)
    active = col.unsqueeze(0) < state.window.unsqueeze(-1)
    sw_sum = (state.sw_ring * active).sum(dim=-1)
    sw_mean = sw_sum / state.sw_count.to(ent.dtype).clamp_min(1.0)
    step_mean = state.step_sum / state.step_len.to(ent.dtype).clamp_min(1.0)
    use_sw = state.step_len < state.window
    sw_valid = state.sw_count > 0
    h_step = torch.where(use_sw, torch.where(sw_valid, sw_mean, ent), step_mean)

    # Running global mean H_bar (falls back to H_step on the first token).
    gn_valid = state.global_n > 0
    global_mean = torch.where(
        gn_valid, state.global_sum / state.global_n.to(ent.dtype).clamp_min(1.0), h_step
    )

    # Eq. 2: an uncertain step raises the threshold to its own entropy. While
    # the sliding window still covers every token seen (global_n <= window),
    # H_step and H_bar are the same set, so the reference compares them equal
    # and never marks the step high; gate on that to avoid a spurious flip
    # from float reduction-order differences.
    high_step = (state.global_n > state.window) & (h_step > global_mean)
    tau_t = torch.where(high_step, h_step, state.tau0)
    # Eq. 1: T_high at/above the threshold, T_low below it.
    reset_temp = torch.where(ent >= tau_t, state.t_high, state.t_low)
    temperature = torch.where(enabled, reset_temp, state.base)

    # Bookkeeping, applied after the decision, for ReSET rows only.
    state.step_sum.copy_(torch.where(enabled, state.step_sum + ent, state.step_sum))
    state.step_len.copy_(torch.where(enabled, state.step_len + 1, state.step_len))
    slot = torch.remainder(state.sw_pos, state.window.to(state.sw_pos.dtype))
    slot = slot.unsqueeze(-1)
    kept = state.sw_ring.gather(1, slot).squeeze(-1)
    state.sw_ring.scatter_(1, slot, torch.where(enabled, ent, kept).unsqueeze(-1))
    state.sw_pos.copy_(torch.where(enabled, state.sw_pos + 1, state.sw_pos))
    state.sw_count.copy_(
        torch.where(
            enabled, torch.minimum(state.sw_count + 1, state.window), state.sw_count
        )
    )
    state.global_sum.copy_(
        torch.where(enabled, state.global_sum + ent, state.global_sum)
    )
    state.global_n.copy_(torch.where(enabled, state.global_n + 1, state.global_n))
    state.prev_was_nl.copy_(
        torch.where(enabled, last_is_nl.to(state.prev_was_nl.dtype), state.prev_was_nl)
    )

    return temperature


def resolve_reset_speculative(
    logits: torch.Tensor,
    base_rows: torch.Tensor,
    chain_lens: torch.Tensor,
    last_committed: torch.Tensor,
    draft_ids: torch.Tensor,
    base_gen_step: torch.Tensor,
    nl_lut: torch.Tensor,
    dnl_lut: torch.Tensor,
    state: ResetState,
    max_chain: int,
) -> list[dict[str, torch.Tensor]]:
    """Resolve ReSET temperatures across speculative draft chains, in place.

    Rejection sampling commits a prefix of each request's draft chain, so the
    temperature at chain position ``j`` must be what sequential ReSET would
    resolve after generating exactly the draft prefix: the target logits row
    at position ``j`` is conditioned on draft tokens ``0..j-1``, which *are*
    the committed tokens wherever the walk reaches ``j``. The per-position
    inputs are therefore all known before acceptance:

    * ``last_token`` at position ``j`` is the previously committed token
      (``j == 0``) or draft token ``j - 1``;
    * ``gen_step`` at position ``j`` is the committed count plus ``j``;
    * the entropy folded into the running state is a property of the target
      distribution at ``j``, not of which token gets sampled there.

    The one thing acceptance decides is where the chain stops, so this scans
    the chain with `resolve_reset`, scales each row by its position's
    temperature in place, and returns a per-position snapshot of the running
    state. The caller commits snapshot ``num_sampled - 1`` once the rejection
    sampler reports how many tokens were actually committed; entropy
    bookkeeping for positions past the stop point is dropped with the rest of
    the snapshot list.

    Args:
        logits: ``[num_rows, vocab]`` processed logits for the whole
            verification batch; rows of ReSET requests are scaled in place.
        base_rows: ``[R]`` row index of each ReSET request's chain position 0.
        chain_lens: ``[R]`` number of logits rows in each request's chain.
        last_committed: ``[R]`` each request's most recent committed token.
            Ignored where ``base_gen_step == 0``.
        draft_ids: ``[R, D]`` proposed draft token ids per request.
        base_gen_step: ``[R]`` committed generated-token count before this
            step.
        nl_lut: ``[vocab]`` bool, true for single-newline token ids.
        dnl_lut: ``[vocab]`` bool, true for double-newline token ids.
        state: Working `ResetState` for the ``R`` ReSET rows; advanced in
            place and left positioned after the full chain (the caller uses
            the snapshots, not this end state).
        max_chain: Host-side upper bound on chain length (the loop bound).

    Returns:
        ``max_chain`` snapshots; entry ``j`` is the running state after
        processing chain position ``j``.
    """
    base_enabled = (state.enabled != 0).to(state.enabled.dtype)
    zero = torch.zeros((), dtype=state.enabled.dtype, device=logits.device)
    snapshots: list[dict[str, torch.Tensor]] = []
    for j in range(max_chain):
        active = (base_enabled != 0) & (j < chain_lens)
        state.enabled.copy_(torch.where(active, base_enabled, zero))
        # Rows past a request's chain end clamp to its own position-0 row;
        # their temperature resolves to the (1.0) base, an exact no-op divide.
        rows_j = torch.where(active, base_rows + j, base_rows)
        if j == 0:
            last_tok_j = last_committed
        else:
            last_tok_j = draft_ids[:, min(j - 1, draft_ids.shape[1] - 1)]
        temperature = resolve_reset(
            logits[rows_j],
            last_tok_j,
            base_gen_step + j,
            nl_lut,
            dnl_lut,
            state,
        )
        logits[rows_j] = logits[rows_j] / temperature.unsqueeze(-1)
        snapshots.append(state.clone_running())
    state.enabled.copy_(base_enabled)
    return snapshots


def make_reset_state(max_num_reqs: int, device: torch.device) -> ResetState:
    """Allocate a full-batch `ResetState` with zeroed running state.

    Configuration rows are filled in per request by the caller; the running
    state starts clean so a freshly (re)admitted row begins its entropy
    history from scratch.
    """

    def zeros(dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(max_num_reqs, dtype=dtype, device=device)

    return ResetState(
        enabled=zeros(torch.int32),
        base=torch.ones(max_num_reqs, dtype=torch.float32, device=device),
        t_low=torch.full((max_num_reqs,), T_LOW, dtype=torch.float32, device=device),
        t_high=torch.full((max_num_reqs,), T_HIGH, dtype=torch.float32, device=device),
        tau0=torch.full((max_num_reqs,), TAU0, dtype=torch.float32, device=device),
        window=torch.full((max_num_reqs,), W, dtype=torch.int32, device=device),
        global_sum=zeros(torch.float32),
        global_n=zeros(torch.int64),
        sw_ring=torch.zeros(max_num_reqs, W, dtype=torch.float32, device=device),
        sw_pos=zeros(torch.int64),
        sw_count=zeros(torch.int64),
        step_sum=zeros(torch.float32),
        step_len=zeros(torch.int64),
        prev_was_nl=zeros(torch.int32),
    )


def get_newline_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Scan the vocabulary for step-boundary token ids.

    Ported from the reference implementation (github.com/aiha-lab/ReSET).

    Returns:
        ``(nl_ids, dnl_ids)`` where ``nl_ids`` decode to exactly ``"\\n"`` and
        ``dnl_ids`` decode to text containing ``"\\n\\n"`` (composite tokens
        such as ``".\\n\\n"`` included). A step boundary is hit when the last
        output token is a ``dnl`` token, or the last two are both ``nl``.
    """
    nl_ids: list[int] = []
    dnl_ids: list[int] = []
    for tid in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if decoded == "\n":
            nl_ids.append(tid)
        if "\n\n" in decoded:
            dnl_ids.append(tid)
    return nl_ids, dnl_ids
