# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max
_COLD_SCAN_BLOCK = 1024

# Fork extension: hesitation-marker penalty. Markers are flattened into one
# token buffer with a CSR-style offset array, so a marker may span several
# tokens without a per-marker padded row.
_MAX_NUM_PENALTY_MARKERS = 64
_MAX_PENALTY_MARKER_TOKENS = 512


class ThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        # Fork extension: absent on a config that carries only the upstream
        # boundary fields, so read it defensively.
        markers = (
            []
            if reasoning_config is None
            else getattr(reasoning_config, "reasoning_marker_token_ids", None) or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        # Read by ``Sampler._requires_logits_processing``; must exist even when
        # reasoning is disabled, since the gate is consulted unconditionally.
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)
        self.use_marker_penalty = np.zeros(self.max_num_reqs, dtype=bool)
        self.use_answer_reserve = np.zeros(self.max_num_reqs, dtype=bool)
        self.reasoning_marker_penalty: UvaBackedTensor | None = None
        self._num_markers = 0
        if not self.enabled:
            return

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()

        # Fork extension: answer reserve. Forces the end marker once the
        # remaining output budget approaches the reserve, so the model always
        # keeps room to write an answer.
        self.reasoning_answer_reserve = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.reasoning_answer_reserve.np.fill(-1)
        self.reasoning_answer_reserve.copy_to_uva()
        self.max_tokens = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.max_tokens.np.fill(-1)
        self.max_tokens.copy_to_uva()
        self._reserve_dirty = False

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
        self._budget_dirty = False

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids, dtype=torch.int32, device=self.device
        )

        self._init_marker_penalty(markers)

    def _init_marker_penalty(self, markers: list[list[int]]) -> None:
        """Flatten configured markers into a token buffer plus CSR offsets.

        Markers of differing lengths share one contiguous token array so the
        penalty kernel can index marker ``m`` as ``tokens[offsets[m]:
        offsets[m + 1]]`` without padding every marker to the longest one.
        """
        flat: list[int] = []
        offsets: list[int] = [0]
        for marker in markers[:_MAX_NUM_PENALTY_MARKERS]:
            if not marker or len(flat) + len(marker) > _MAX_PENALTY_MARKER_TOKENS:
                continue
            flat.extend(marker)
            offsets.append(len(flat))

        self._num_markers = len(offsets) - 1
        if self._num_markers == 0:
            return

        self._marker_tokens = torch.tensor(flat, dtype=torch.int32, device=self.device)
        self._marker_offsets = torch.tensor(
            offsets, dtype=torch.int32, device=self.device
        )
        self.reasoning_marker_penalty = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.float32
        )
        self.reasoning_marker_penalty.np.fill(0.0)
        self.reasoning_marker_penalty.copy_to_uva()
        self._penalty_dirty = False

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True

        reserve = sampling_params.reasoning_answer_reserve
        max_tokens = sampling_params.max_tokens
        # A reserve without max_tokens has no output budget to reserve from.
        active = reserve is not None and reserve > 0 and max_tokens is not None
        self.use_answer_reserve[req_idx] = active
        reserve_val = reserve if active else -1
        max_tokens_val = max_tokens if active else -1
        if (
            self.reasoning_answer_reserve.np[req_idx] != reserve_val
            or self.max_tokens.np[req_idx] != max_tokens_val
        ):
            self.reasoning_answer_reserve.np[req_idx] = reserve_val
            self.max_tokens.np[req_idx] = max_tokens_val
            self._reserve_dirty = True
        if active and req_idx not in self._reset_reqs:
            self._reset_reqs.append(req_idx)

        if self.reasoning_marker_penalty is not None:
            penalty = sampling_params.reasoning_marker_penalty or 0.0
            self.use_marker_penalty[req_idx] = penalty != 0.0
            if self.reasoning_marker_penalty.np[req_idx] != penalty:
                self.reasoning_marker_penalty.np[req_idx] = penalty
                self._penalty_dirty = True
            if penalty != 0.0 and req_idx not in self._reset_reqs:
                # Marker penalty needs the committed marker cache, which the
                # cold-scan kernel only maintains for reset requests.
                self._reset_reqs.append(req_idx)

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
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False
        if self.reasoning_marker_penalty is not None and self._penalty_dirty:
            self.reasoning_marker_penalty.copy_to_uva()
            self._penalty_dirty = False
        if self._reserve_dirty:
            self.reasoning_answer_reserve.copy_to_uva()
            self.max_tokens.copy_to_uva()
            self._reserve_dirty = False

    @property
    def tracked_np(self) -> np.ndarray:
        """Per-request mask of requests with any reasoning control set.

        Read by ``Sampler._requires_logits_processing`` so a request using only
        reasoning controls -- greedy or temperature 1.0, no penalties, no bad
        words -- still enters the logits-processing path instead of returning
        early and leaving the budget silently inert.
        """
        return (
            self.use_thinking_budget | self.use_marker_penalty | self.use_answer_reserve
        )

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        penalty_state = self.reasoning_marker_penalty
        use_budget = np.any(self.use_thinking_budget[idx_mapping_np])
        use_penalty = penalty_state is not None and np.any(
            self.use_marker_penalty[idx_mapping_np]
        )
        use_reserve = np.any(self.use_answer_reserve[idx_mapping_np])
        if not use_budget and not use_penalty and not use_reserve:
            return

        apply_thinking_budget(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
            marker_tokens=self._marker_tokens if use_penalty else None,
            marker_offsets=self._marker_offsets if use_penalty else None,
            marker_penalty=(
                penalty_state.gpu if use_penalty and penalty_state is not None else None
            ),
            num_markers=self._num_markers if use_penalty else 0,
            answer_reserve=self.reasoning_answer_reserve.gpu if use_reserve else None,
            max_tokens=self.max_tokens.gpu if use_reserve else None,
            prompt_len=self.req_states.prompt_len.gpu if use_reserve else None,
        )


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    marker_penalty_ptr,
    answer_reserve_ptr,
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
    HAS_PENALTY: tl.constexpr,
    HAS_RESERVE: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    # The marker penalty and answer reserve read this cache too, so a request
    # using either without a budget still needs it maintained.
    needs_cache = budget >= 0
    if HAS_PENALTY:
        needs_cache = needs_cache or tl.load(marker_penalty_ptr + req_state_idx) != 0.0
    if HAS_RESERVE:
        needs_cache = needs_cache or tl.load(answer_reserve_ptr + req_state_idx) > 0
    if not needs_cache:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the first
        # block with a marker; only the relative order of the two positions
        # found matters below.
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
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
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
    reasoning_end_token_ids_ptr,
    answer_reserve_ptr,
    max_tokens_ptr,
    prompt_len_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    HAS_RESERVE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    reserve = -1
    if HAS_RESERVE:
        reserve = tl.load(answer_reserve_ptr + req_state_idx)
    if budget < 0 and reserve <= 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

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

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    budget_exhausted = budget >= 0 and num_reasoning_tokens >= budget

    # The answer reserve is an independent trigger for the same forcing: end
    # thinking once the remaining output budget has only the reserve left.
    reserve_reached = False
    if HAS_RESERVE and reserve > 0:
        max_tokens = tl.load(max_tokens_ptr + req_state_idx)
        if max_tokens > 0:
            prompt_len = tl.load(prompt_len_ptr + req_state_idx)
            produced = effective_len - prompt_len
            reserve_reached = max_tokens - reserve - produced <= 0

    if not budget_exhausted and not reserve_reached:
        return

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


@triton.jit
def _marker_penalty_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    marker_tokens_ptr,
    marker_offsets_ptr,
    marker_penalty_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    START_LEN: tl.constexpr,
):
    """Subtract a penalty from markers that would complete inside a think block.

    One program per (logit row, marker). Mirrors ``_bad_words_kernel``: match
    every marker token but the last against the request's recent history, then
    act on the final token. Unlike bad words the final token is penalised
    rather than banned, so a shared prefix is not discouraged on its own.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    marker_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    penalty = tl.load(marker_penalty_ptr + req_state_idx)
    if penalty == 0.0:
        return

    # Penalise only inside a reasoning block, using the same cached boundaries
    # the budget kernel maintains: a start marker seen after the last end.
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)
    if last_start < 0 or last_start <= last_end:
        return

    start = tl.load(marker_offsets_ptr + marker_idx)
    end = tl.load(marker_offsets_ptr + marker_idx + 1)
    if end <= start:
        return
    last_token = tl.load(marker_tokens_ptr + end - 1)
    if last_token < 0 or last_token >= vocab_size:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    # The marker may only complete where its preceding tokens already match.
    prefix_len = end - 1 - start
    reasoning_start = last_start + START_LEN
    if effective_len - prefix_len < reasoning_start:
        return

    match = True
    for i in range(prefix_len):
        expected = tl.load(marker_tokens_ptr + start + i)
        actual = _load_effective_token(
            all_token_ids_ptr,
            all_token_ids_stride,
            input_ids_ptr,
            cur_req_first_pos,
            req_state_idx,
            total_len,
            effective_len - prefix_len + i,
        )
        match = match & (actual == expected)

    if match:
        offset = token_idx * logits_stride + last_token
        tl.store(logits_ptr + offset, tl.load(logits_ptr + offset) - penalty)


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
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
    reasoning_end_token_ids: torch.Tensor,
    marker_tokens: torch.Tensor | None = None,
    marker_offsets: torch.Tensor | None = None,
    marker_penalty: torch.Tensor | None = None,
    num_markers: int = 0,
    answer_reserve: torch.Tensor | None = None,
    max_tokens: torch.Tensor | None = None,
    prompt_len: torch.Tensor | None = None,
) -> None:
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

    _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        thinking_token_budget,
        # Unused when no penalty is configured; pass a valid pointer anyway.
        marker_penalty if marker_penalty is not None else thinking_token_budget,
        answer_reserve if answer_reserve is not None else thinking_token_budget,
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
        HAS_PENALTY=marker_penalty is not None,
        HAS_RESERVE=answer_reserve is not None,
    )

    # The penalty reads the marker cache the kernel above just refreshed, and
    # must run before forcing so a forced end token is never penalised.
    if num_markers > 0:
        assert marker_tokens is not None
        assert marker_offsets is not None
        assert marker_penalty is not None
        _marker_penalty_kernel[(num_tokens, num_markers)](
            logits,
            logits.stride(0),
            logits.shape[1],
            expanded_idx_mapping,
            marker_tokens,
            marker_offsets,
            marker_penalty,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            input_ids,
            expanded_local_pos,
            cached_last_start,
            cached_last_end,
            START_LEN=start_len,
        )

    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
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
        reasoning_end_token_ids,
        # Unused when no reserve is configured; pass valid pointers anyway.
        answer_reserve if answer_reserve is not None else thinking_token_budget,
        max_tokens if max_tokens is not None else thinking_token_budget,
        prompt_len if prompt_len is not None else thinking_token_budget,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        HAS_RESERVE=answer_reserve is not None,
    )
