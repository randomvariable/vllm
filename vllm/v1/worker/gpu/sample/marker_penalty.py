# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side overthinking-marker logit penalty (arXiv 2606.00206).

``Quantized Reasoning Models Think They Need to Think Longer, but They Do Not``
(Lotfi et al., arXiv 2606.00206) shows that post-training-quantized reasoning
models over-sample leading-space "branch-opening" tokens (hesitation /
redirection markers such as ``_wait``, ``_But``, ``_alternatively``) at
high-entropy decoding positions, and that subtracting a fixed logit penalty
from a curated marker set measurably shortens chain-of-thought while
preserving or improving accuracy.

This module applies that penalty as a per-request, opt-in reasoning-phase
logits step in the V2 model runner's sampler. It is intentionally independent
of the thinking token budget: a request may enable the marker penalty without
any budget, so the reasoning-boundary cache it needs is maintained on the
penalty state itself rather than piggybacking on ``ThinkingBudgetState``,
whose boundary cache is only refreshed while a *budgeted* request is in the
batch.

Composition
-----------
`Sampler.apply_sampling_params` applies logits processors in order: logit bias
-> penalties -> bad words -> thinking-budget forcing -> ReSET entropy
temperature -> static/phase temperature -> min-p -> top-k/top-p. The marker
penalty is inserted immediately after thinking-budget forcing and before
ReSET, and therefore runs on the same post-processing logits ReSET already
observes: the two are *mechanism*-additive but *outcome*-coupled, because the
penalty subtracts mass from ~50 tokens, lowering the entropy ReSET measures
and biasing it toward ``T_low``. They must be calibrated as one stack, never
read as two separable knobs.

The penalty is applied only while the request is inside its reasoning block
and its thinking budget is not yet exhausted (so budget-forced rows are left
untouched), at every decode position - including speculative draft positions,
where the phase is derived per position by conditioning on the committed
markers (from the cache) plus the draft prefix. Requests with no marker
penalty take the existing fast path with no new kernel launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.thinking_budget import (
    _COLD_SCAN_BLOCK,
    _load_effective_token,
)
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig


@triton.jit
def _update_marker_boundary_cache_kernel(
    req_ids_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the
        # first block with a marker; only the relative order of the two
        # positions found matters below.
        block_hi = total_len
        while block_hi > 0 and last_start < 0 and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            block_hi = block_lo
    else:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _marker_penalty_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    penalty_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    marker_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MARKER_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    penalty = tl.load(penalty_ptr + req_state_idx)
    if penalty <= 0.0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    # Reasoning-phase detection, per position including draft positions: the
    # cache holds the newest committed start/end markers; the window scan
    # `[total_len - START_LEN + 1, effective_len)` extends that with tokens
    # committed this step plus the draft prefix. At chain position j the draft
    # tokens 0..j-1 are the committed tokens in every branch that reaches j,
    # so conditioning on them is sequential-equivalent, not speculative
    # contamination. Positions past the acceptance point are discarded with
    # the rest of the chain.
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    start_lo = total_len - START_LEN + 1
    if start_lo < 0:
        start_lo = 0
    for i in tl.range(start_lo, effective_len - START_LEN + 1):
        start_match = True
        for j in tl.static_range(0, START_LEN):
            expected = tl.load(reasoning_start_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            start_match = start_match & (actual == expected)
        if start_match:
            last_start = i

    end_lo = total_len - NATURAL_END_LEN + 1
    if end_lo < 0:
        end_lo = 0
    for i in tl.range(end_lo, effective_len - NATURAL_END_LEN + 1):
        end_match = True
        for j in tl.static_range(0, NATURAL_END_LEN):
            expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            end_match = end_match & (actual == expected)
        if end_match:
            last_end = i

    # Not inside the reasoning block: never penalize answer-phase or
    # non-reasoning output.
    if last_start < 0 or last_start <= last_end:
        return

    # Never penalize a row whose thinking budget is already exhausted and is
    # being force-driven toward the reasoning end marker. Replicate the budget
    # kernel's exhaustion condition so budget-forced rows are left untouched.
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget >= 0:
        reasoning_start = last_start + START_LEN
        num_reasoning_tokens = effective_len - reasoning_start
        if num_reasoning_tokens >= budget:
            return

    for j in tl.static_range(0, MARKER_LEN):
        marker_tid = tl.load(marker_token_ids_ptr + j)
        tl.store(
            logits_ptr + token_idx * logits_stride + marker_tid,
            tl.load(logits_ptr + token_idx * logits_stride + marker_tid) - penalty,
        )


def apply_marker_penalty(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    penalty: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    marker_token_ids: torch.Tensor,
) -> None:
    """Refresh the marker boundary cache and apply the marker penalty.

    Launched only when at least one row in the batch enables the marker
    penalty (the caller gates on that). The cache refresh scans each
    marker-enabled request's committed chain for the newest reasoning
    start/end markers; the penalty then applies to rows still inside their
    reasoning block.
    """
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    marker_len = marker_token_ids.shape[0]

    _update_marker_boundary_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _marker_penalty_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        penalty,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        marker_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MARKER_LEN=marker_len,
    )


class ReasoningMarkerPenaltyState:
    """Model Runner V2 state for the overthinking-marker logit penalty.

    Holds a per-request penalty and enabled bit in UVA-backed arrays, one
    model-wide marker-token LUT on device, and its own committed
    reasoning-boundary cache (``cached_last_start`` / ``cached_last_end`` /
    ``cached_scan_pos``) so the reasoning phase is observable independently of
    the thinking token budget. Disabled requests and disabled batches take the
    existing fast path with no new kernel launch.
    """

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: ReasoningConfig | None,
        marker_token_ids: list[int],
        thinking_token_budget: torch.Tensor | None,
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        # Enabled only when the reasoning boundary is observable AND the active
        # tokenizer resolved at least one marker. Fail closed otherwise.
        self.enabled = bool(start_ids and natural_end_ids and marker_token_ids)
        # Defined regardless of `enabled`: the sampler's needs_logits_processing
        # aggregation reads this for every request.
        self.use_marker_penalty = np.zeros(self.max_num_reqs, dtype=bool)
        if not self.enabled:
            return
        # Thinking-token-budget UVA tensor, used only to reject budget-exhausted
        # (force-driven) rows. Reasoning is enabled whenever markers resolve, so
        # the owning ThinkingBudgetState always provides one.
        assert thinking_token_budget is not None
        self._budget_gpu = thinking_token_budget

        self.penalty = UvaBackedTensor(self.max_num_reqs, dtype=torch.float32)
        self.penalty.np.fill(0.0)
        self.penalty.copy_to_uva()
        self._penalty_dirty = False

        self.cached_last_start = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_last_end = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_scan_pos = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._reset_reqs: list[int] = []

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.marker_token_ids = torch.tensor(
            marker_token_ids, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        penalty = sampling_params.reasoning_marker_penalty
        self.use_marker_penalty[req_idx] = penalty is not None and penalty > 0.0
        if penalty is None:
            penalty = 0.0
        if penalty > 0.0:
            # Fresh request: reset its boundary cache so the phase is derived
            # from its own prompt, not a recycled cache row.
            self._reset_reqs.append(req_idx)
        if self.penalty.np[req_idx] != penalty:
            self.penalty.np[req_idx] = penalty
            self._penalty_dirty = True

    def apply_staged_writes(self) -> None:
        if not self.enabled:
            return
        if self._reset_reqs:
            idx = async_tensor_h2d(
                self._reset_reqs, dtype=torch.int64, device=self.device
            )
            self.cached_last_start.index_fill_(0, idx, -1)
            self.cached_last_end.index_fill_(0, idx, -1)
            self.cached_scan_pos.index_fill_(0, idx, 0)
            self._reset_reqs.clear()
        if self._penalty_dirty:
            self.penalty.copy_to_uva()
            self._penalty_dirty = False

    def reset_cache(self, idx: torch.Tensor) -> None:
        """Reset the boundary cache for the given request indices."""
        self.cached_last_start.index_fill_(0, idx, -1)
        self.cached_last_end.index_fill_(0, idx, -1)
        self.cached_scan_pos.index_fill_(0, idx, 0)

    def apply(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.enabled or not np.any(self.use_marker_penalty[idx_mapping_np]):
            return

        apply_marker_penalty(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            self.penalty.gpu,
            self._budget_gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.marker_token_ids,
        )
