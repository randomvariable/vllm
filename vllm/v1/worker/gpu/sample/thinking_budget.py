# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""Tensor-native ThinkingBudgetState for Model Runner V2.

Enforces three per-request reasoning controls by modifying logits on the GPU
without any device-to-host token-id synchronization:

1. ``thinking_token_budget`` — caps tokens inside ``<think>...</think>``, then
   forces the end-of-thinking token.
2. ``reasoning_answer_reserve`` — forces end-of-thinking when remaining output
   budget approaches a reserve threshold so the model always has room to emit
   an answer.
3. ``reasoning_marker_penalty`` — penalises hesitation marker tokens inside
   reasoning blocks.

Modeled on :class:`BadWordsState` (one Triton kernel, grid ``(num_reqs,)``,
delta-scanning committed+spec tokens with KMP partial-match carry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

# Maximum supported length of multi-token think start/end marker sequences.
_MAX_MARKER_LEN = 8
# Maximum number of marker-penalty tokens tracked per request.
_MAX_MARKER_TOKENS = 64


class ThinkingBudgetState:
    """Per-request reasoning-control state, applied on-device at sample time.

    Follows the :class:`BadWordsState` lifecycle:
    - ``add_request`` stages per-request config + prompt scan
    - ``apply_staged_writes`` syncs staged CPU data to GPU
    - ``apply_thinking_budget`` launches the kernel from the sampler hot path
    """

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: ReasoningConfig | None,
        num_speculative_tokens: int = 0,
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
        self.num_speculative_tokens = num_speculative_tokens

        self._enabled = reasoning_config is not None and reasoning_config.enabled
        # Allocated before the disabled early-return: Sampler's per-step gate
        # reads this mask unconditionally.
        self._has_tracked = np.zeros(self.max_num_reqs, dtype=bool)
        if not self._enabled:
            return

        # --- Marker token IDs (shared across all requests) ---
        start_ids = (
            reasoning_config.reasoning_start_token_ids
            if reasoning_config and reasoning_config.reasoning_start_token_ids
            else []
        )
        end_ids = (
            reasoning_config.reasoning_end_token_ids
            if reasoning_config and reasoning_config.reasoning_end_token_ids
            else []
        )
        marker_ids = (
            reasoning_config.reasoning_marker_token_ids
            if reasoning_config and reasoning_config.reasoning_marker_token_ids
            else []
        )

        # Pad marker sequences to _MAX_MARKER_LEN for fixed-shape kernel args.
        self._start_len = len(start_ids)
        self._end_len = len(end_ids)
        self._start_padded = self._pad(start_ids, _MAX_MARKER_LEN)
        self._end_padded = self._pad(end_ids, _MAX_MARKER_LEN)

        # Flatten marker-penalty tokens for the kernel (fixed-size buffer).
        self._num_markers = min(len(marker_ids), _MAX_MARKER_TOKENS)
        self._markers_padded = self._pad(
            marker_ids[: self._num_markers], _MAX_MARKER_TOKENS
        )

        # --- Per-request config (staged on add_request, synced to GPU) ---
        self.thinking_token_budget = StagedWriteTensor(
            (self.max_num_reqs,), dtype=torch.int32, device=self.device
        )
        self.reasoning_answer_reserve = StagedWriteTensor(
            (self.max_num_reqs,), dtype=torch.int32, device=self.device
        )
        self.max_tokens = StagedWriteTensor(
            (self.max_num_reqs,), dtype=torch.int32, device=self.device
        )
        self.reasoning_marker_penalty = StagedWriteTensor(
            (self.max_num_reqs,), dtype=torch.float32, device=self.device
        )
        # -1 = no budget/reserve set (0 is a valid budget meaning "no thinking").
        # These are StagedWriteTensors: initialize the device buffer directly.
        # ``StagedWriteTensor`` exposes no ``.np`` view (UvaBackedTensor does);
        # per-request values land via ``stage_write_elem`` + ``apply_write``.
        self.thinking_token_budget.gpu.fill_(-1)
        self.reasoning_answer_reserve.gpu.fill_(-1)
        self.max_tokens.gpu.fill_(-1)
        self.reasoning_marker_penalty.gpu.fill_(0.0)

        # --- Per-request running state (GPU-resident, updated by kernel) ---
        # These MUST NOT be UvaBackedTensor. That type's ``.np`` host array is
        # not aliased to ``.gpu``: ``copy_to_uva()`` copies host -> a pooled
        # device buffer and rebinds ``.gpu``. For kernel-written state that
        # would leave every host read stale, and would clobber the kernel's
        # accumulated device state on the next flush, resetting in-flight
        # requests to their prompt-scan values. Every other UvaBackedTensor in
        # the tree holds host-authored config, which is why the type is right
        # there and wrong here.
        #
        # StagedWriteTensor has the shape this needs: the device tensor is the
        # single source of truth and host writes are applied per-row, touching
        # only the rows actually staged.
        def _state_tensor(dtype: torch.dtype) -> StagedWriteTensor:
            return StagedWriteTensor(
                (self.max_num_reqs,), dtype=dtype, device=self.device
            )

        # bool is not a StagedWriteTensor dtype; flags are int32 0/1.
        self.in_think = _state_tensor(torch.int32)
        self.in_end = _state_tensor(torch.int32)
        self.think_count = _state_tensor(torch.int32)
        self.countdown = _state_tensor(torch.int32)
        self.end_count = _state_tensor(torch.int32)
        self.seen_len = _state_tensor(torch.int32)  # last-scanned total_len
        self.kmp_start = _state_tensor(torch.int32)  # KMP progress for <think>
        self.kmp_end = _state_tensor(torch.int32)  # KMP progress for </think>

        # Force output (written by kernel, read back by the Python scatter).
        self.force_active = _state_tensor(torch.int32)
        self.force_offset = _state_tensor(torch.int32)
        self.force_end_count = _state_tensor(torch.int32)
        self.countdown.gpu.fill_(-1)

        # CPU mirror for fast gating (avoids GPU readback in hot path).
        # CPU mirror of the marker penalty. The device copy lives in a
        # StagedWriteTensor, which exposes no ``.np`` view, and the host path
        # must not read back from the device to decide whether to do work.
        self._marker_penalty_cpu = np.zeros(self.max_num_reqs, dtype=np.float32)
        # CPU mirror of in_think at add_request time. NOTE: this reflects the
        # prompt scan only -- the kernel's live updates stay on device. It is
        # therefore usable as a cheap "could this request ever be in think
        # mode" pre-filter, never as the authoritative gate.
        self._in_think_cpu = np.zeros(self.max_num_reqs, dtype=bool)

        # Upload constant marker tensors to device.
        self._start_gpu = torch.tensor(
            self._start_padded, dtype=torch.int32, device=self.device
        )
        self._end_gpu = torch.tensor(
            self._end_padded, dtype=torch.int32, device=self.device
        )
        self._markers_gpu = torch.tensor(
            self._markers_padded, dtype=torch.int32, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pad(ids: list[int], length: int, fill: int = -1) -> list[int]:
        """Pad a token-id list to fixed length with sentinel -1."""
        result = list(ids[:length])
        result.extend([fill] * (length - len(result)))
        return result

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        all_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        """Stage per-request config and perform initial prompt scan."""
        if not self._enabled:
            return

        budget = sampling_params.thinking_token_budget
        reserve = sampling_params.reasoning_answer_reserve
        marker_pen = sampling_params.reasoning_marker_penalty
        mt = sampling_params.max_tokens

        self.thinking_token_budget.stage_write_elem(
            req_idx, budget if budget is not None else -1
        )
        self.reasoning_answer_reserve.stage_write_elem(
            req_idx, reserve if reserve is not None else -1
        )
        self.max_tokens.stage_write_elem(req_idx, mt if mt is not None else -1)
        self.reasoning_marker_penalty.stage_write_elem(
            req_idx, marker_pen if marker_pen is not None else 0.0
        )
        self._marker_penalty_cpu[req_idx] = (
            marker_pen if marker_pen is not None else 0.0
        )

        tracked = (
            budget is not None
            or reserve is not None
            or (marker_pen is not None and marker_pen != 0.0)
        )
        self._has_tracked[req_idx] = tracked

        # Initial prompt scan: detect if prompt ends inside a <think> block.
        in_think = False
        think_count = 0
        countdown = budget if budget is not None else -1
        kmp_start = 0
        kmp_end = 0
        seen_len = len(all_token_ids)

        if tracked and self._start_len > 0:
            # Scan prompt for last complete <think> and </think>.
            last_start = self._find_last_match(
                all_token_ids, self._start_padded[: self._start_len]
            )
            last_end = self._find_last_match(
                all_token_ids, self._end_padded[: self._end_len]
            )
            if last_start >= 0 and (last_end < 0 or last_start > last_end):
                in_think = True
                marker_tail = last_start + self._start_len
                think_count = max(0, len(all_token_ids) - marker_tail)
                if budget is not None:
                    countdown = max(0, budget - think_count)

        self.in_think.stage_write_elem(req_idx, int(in_think))
        self.in_end.stage_write_elem(req_idx, 0)
        self.think_count.stage_write_elem(req_idx, think_count)
        self.countdown.stage_write_elem(req_idx, countdown)
        self.end_count.stage_write_elem(req_idx, 0)
        self.seen_len.stage_write_elem(req_idx, seen_len)
        self.kmp_start.stage_write_elem(req_idx, kmp_start)
        self.kmp_end.stage_write_elem(req_idx, kmp_end)
        self.force_active.stage_write_elem(req_idx, 0)
        self.force_offset.stage_write_elem(req_idx, 0)
        self.force_end_count.stage_write_elem(req_idx, 0)
        self._in_think_cpu[req_idx] = in_think

    @staticmethod
    def _find_last_match(tokens: list[int], pattern: list[int]) -> int:
        """Return the start index of the last occurrence of pattern, or -1."""
        if not pattern:
            return -1
        plen = len(pattern)
        for i in range(len(tokens) - plen, -1, -1):
            if tokens[i : i + plen] == pattern:
                return i
        return -1

    def remove_request(self, req_idx: int) -> None:
        """Clear per-request state when a slot is recycled."""
        if not self._enabled:
            return
        self._has_tracked[req_idx] = False
        # Config tensors are StagedWriteTensors: no ``.np`` view, so clear them
        # through the same staged-write path add_request uses.
        self.thinking_token_budget.stage_write_elem(req_idx, -1)
        self.reasoning_answer_reserve.stage_write_elem(req_idx, -1)
        self.max_tokens.stage_write_elem(req_idx, -1)
        self.reasoning_marker_penalty.stage_write_elem(req_idx, 0.0)
        self._marker_penalty_cpu[req_idx] = 0.0
        self.in_think.stage_write_elem(req_idx, 0)
        self.in_end.stage_write_elem(req_idx, 0)
        self.think_count.stage_write_elem(req_idx, 0)
        self.countdown.stage_write_elem(req_idx, -1)
        self.end_count.stage_write_elem(req_idx, 0)
        self.seen_len.stage_write_elem(req_idx, 0)
        self.force_active.stage_write_elem(req_idx, 0)
        # Reset KMP partial-match and force-output state too: a recycled
        # req_idx must not inherit mid-marker progress from a prior occupant,
        # or a stale partial think match could spuriously toggle think mode.
        self.kmp_start.stage_write_elem(req_idx, 0)
        self.kmp_end.stage_write_elem(req_idx, 0)
        self.force_offset.stage_write_elem(req_idx, 0)
        self.force_end_count.stage_write_elem(req_idx, 0)
        self._in_think_cpu[req_idx] = False

    def apply_staged_writes(self) -> None:
        """Flush staged per-request writes to the device tensors.

        Every tensor here is a StagedWriteTensor, so this applies only the rows
        actually staged this step and leaves kernel-accumulated state for
        in-flight requests untouched.
        """
        if not self._enabled:
            return
        for t in (
            self.thinking_token_budget,
            self.reasoning_answer_reserve,
            self.max_tokens,
            self.reasoning_marker_penalty,
            self.in_think,
            self.in_end,
            self.think_count,
            self.countdown,
            self.end_count,
            self.seen_len,
            self.kmp_start,
            self.kmp_end,
            self.force_active,
            self.force_offset,
            self.force_end_count,
        ):
            t.apply_write()

    @property
    def tracked_np(self) -> np.ndarray:
        """Per-request bool mask of requests with any reasoning control set.

        Read by ``Sampler._requires_logits_processing`` so that a request
        using only reasoning controls -- greedy or temperature 1.0, no
        penalties, no bad words -- still enters the logits-processing path.
        """
        return self._has_tracked

    @property
    def has_tracked_requests(self) -> bool:
        """True when any active request has reasoning controls set."""
        if not self._enabled:
            return False
        return bool(self._has_tracked.any())

    # ------------------------------------------------------------------ #
    # Hot path
    # ------------------------------------------------------------------ #

    def apply_thinking_budget(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-request think state on-device and apply logits forcing.

        Called from :meth:`Sampler.__call__` after penalties, before sampling.
        Equivalent to BadWordsState.apply_bad_words in the call chain.
        """
        if not self._enabled or not self.has_tracked_requests:
            return logits

        num_reqs = int(idx_mapping_np.shape[0])

        # Phase 1: per-request state update kernel.
        _thinking_budget_update_kernel[(num_reqs,)](
            # Token history
            self.req_states.all_token_ids.gpu,
            self.req_states.all_token_ids.gpu.stride(0),
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            # Draft tokens (spec decode)
            self.req_states.draft_tokens,
            self.num_speculative_tokens,
            # Config
            self.thinking_token_budget.gpu,
            self.reasoning_answer_reserve.gpu,
            self.max_tokens.gpu,
            self.reasoning_marker_penalty.gpu,
            # Marker patterns
            self._start_gpu,
            self._end_gpu,
            self._markers_gpu,
            self._start_len,
            self._end_len,
            self._num_markers,
            # State (in/out)
            self.in_think.gpu,
            self.in_end.gpu,
            self.think_count.gpu,
            self.countdown.gpu,
            self.end_count.gpu,
            self.seen_len.gpu,
            self.kmp_start.gpu,
            self.kmp_end.gpu,
            # Force output
            self.force_active.gpu,
            self.force_offset.gpu,
            self.force_end_count.gpu,
            num_reqs=num_reqs,
        )

        # The kernel just wrote in_think and the force triple on device. Both
        # host-side phases below need those values, so take ONE explicit D2H
        # copy rather than four implicit ones. This is a real synchronisation
        # point; Phase 2 removes it by moving the marker penalty into a kernel.
        state_np = (
            torch.stack(
                (
                    self.in_think.gpu,
                    self.force_active.gpu,
                    self.force_offset.gpu,
                    self.force_end_count.gpu,
                )
            )
            .cpu()
            .numpy()
        )
        in_think_np, force_active_np, force_offset_np, force_end_count_np = state_np

        # Phase 2: apply marker penalty (scatter-subtract on penalised rows).
        if self._num_markers > 0:
            self._apply_marker_penalty(
                logits, expanded_idx_mapping, idx_mapping_np, in_think_np
            )

        # Phase 3: apply forcing (set forced end-token to 1e9, mask row).
        self._apply_forcing(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            force_active_np,
            force_offset_np,
            force_end_count_np,
        )

        return logits

    def _apply_marker_penalty(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        in_think_np: np.ndarray,
    ) -> None:
        """Scatter-subtract marker penalty on hesitation tokens inside <think>."""
        # Build CPU index lists (small: only tracked + in_think requests).
        active_rows: list[int] = []
        active_tokens: list[int] = []
        penalties: list[float] = []

        marker_ids = [t for t in self._markers_padded if t >= 0]
        if not marker_ids:
            return

        for req_idx in idx_mapping_np:
            if not self._has_tracked[req_idx]:
                continue
            if not bool(in_think_np[req_idx]):
                continue
            pen = float(self._marker_penalty_cpu[req_idx])
            if pen == 0.0:
                continue
            # Find this request's logit rows via expanded_idx_mapping.
            row_mask = expanded_idx_mapping.eq(req_idx)
            rows = row_mask.nonzero(as_tuple=True)[0].tolist()
            for row in rows:
                for tok in marker_ids:
                    if 0 <= tok < logits.shape[1]:
                        active_rows.append(row)
                        active_tokens.append(tok)
                        penalties.append(pen)

        if not active_rows:
            return

        rows_t = torch.tensor(active_rows, device=logits.device)
        tokens_t = torch.tensor(active_tokens, device=logits.device)
        pens_t = torch.tensor(penalties, dtype=logits.dtype, device=logits.device)
        logits.index_put_((rows_t, tokens_t), -pens_t, accumulate=True)

    def _apply_forcing(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        force_active: np.ndarray,
        force_offset: np.ndarray,
        force_end_count: np.ndarray,
    ) -> None:
        """Force end-of-thinking tokens on requests whose budget is exhausted.

        For each forced request, select the correct logit row (based on
        force_offset within the request's row span) and set it to one-hot
        on the end-token.
        """
        active_rows: list[int] = []
        force_tokens: list[int] = []

        end_ids = [t for t in self._end_padded if t >= 0]

        for req_idx in idx_mapping_np:
            if not force_active[req_idx]:
                continue

            # Map force_offset to absolute logit row.
            row_mask = expanded_idx_mapping.eq(req_idx)
            rows = row_mask.nonzero(as_tuple=True)[0]
            num_rows = rows.shape[0]
            if num_rows == 0:
                continue

            offset = int(force_offset[req_idx])
            offset = max(0, min(offset, num_rows - 1))
            abs_row = int(rows[offset].item())

            # Which end-token to force (multi-token marker continuation).
            ec = int(force_end_count[req_idx])
            if ec < len(end_ids) and end_ids[ec] >= 0:
                active_rows.append(abs_row)
                force_tokens.append(end_ids[ec])

        if not active_rows:
            return

        rows_t = torch.tensor(active_rows, device=logits.device)
        tokens_t = torch.tensor(force_tokens, device=logits.device)

        # Mask entire row to -inf, then force the end-token to +1e9.
        logits[rows_t] = float("-inf")
        logits[rows_t, tokens_t] = 1e9


# ---------------------------------------------------------------------- #
# Triton kernel: per-request state machine
# ---------------------------------------------------------------------- #


@triton.jit
def _thinking_budget_update_kernel(
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    draft_tokens_ptr,
    num_speculative_tokens,
    # Config
    budget_ptr,
    reserve_ptr,
    max_tokens_ptr,
    marker_penalty_ptr,
    # Marker patterns (device constants)
    start_ids_ptr,
    end_ids_ptr,
    markers_ptr,
    start_len,
    end_len,
    num_markers,
    # State (in/out)
    in_think_ptr,
    in_end_ptr,
    think_count_ptr,
    countdown_ptr,
    end_count_ptr,
    seen_len_ptr,
    kmp_start_ptr,
    kmp_end_ptr,
    # Force output
    force_active_ptr,
    force_offset_ptr,
    force_end_count_ptr,
    num_reqs: tl.constexpr,
    max_marker_len: tl.constexpr = _MAX_MARKER_LEN,
):
    """One program per request: delta-scan new tokens, update state, compute force.

    The kernel reads committed tokens from all_token_ids[req, prompt_len:total_len]
    that were added since seen_len, plus draft_tokens for spec-decode. It runs a
    simplified state machine:
    1. Detect <think> start / </think> end via KMP partial-match.
    2. Track think_count and countdown.
    3. When countdown <= 0 or reserve threshold hit: set force_active + force_offset.
    """
    req_idx = tl.program_id(0).to(tl.int64)
    if req_idx >= num_reqs:
        return

    budget = tl.load(budget_ptr + req_idx)
    has_budget = budget >= 0  # -1 means unset

    # If no budget and no reserve, nothing to do (marker penalty handled
    # separately in Python scatter).
    reserve = tl.load(reserve_ptr + req_idx)
    has_reserve = reserve > 0
    marker_pen = tl.load(marker_penalty_ptr + req_idx)
    has_marker = marker_pen != 0.0

    if not (has_budget or has_reserve or has_marker):
        tl.store(force_active_ptr + req_idx, 0)
        return

    prompt_len = tl.load(prompt_len_ptr + req_idx)
    total_len = tl.load(total_len_ptr + req_idx)
    seen_len = tl.load(seen_len_ptr + req_idx)

    # Load current state.
    in_think = tl.load(in_think_ptr + req_idx)
    in_end = tl.load(in_end_ptr + req_idx)
    think_count = tl.load(think_count_ptr + req_idx)
    countdown = tl.load(countdown_ptr + req_idx)
    end_count = tl.load(end_count_ptr + req_idx)
    kmp_s = tl.load(kmp_start_ptr + req_idx)
    kmp_e = tl.load(kmp_end_ptr + req_idx)

    output_len = total_len - prompt_len

    # --- Delta scan: process tokens [seen_len, total_len) ---
    # These are newly committed tokens since the last step.
    for pos in range(seen_len, total_len):
        token = tl.load(all_token_ids_ptr + req_idx * all_token_ids_stride + pos)

        if start_len > 0:
            # KMP advance for <think> start marker.
            expected = tl.load(start_ids_ptr + kmp_s)
            if token == expected:
                kmp_s = kmp_s + 1
                if kmp_s >= start_len:
                    # Complete <think> match — entering think mode.
                    in_think = True
                    marker_tail = pos + 1
                    think_count = marker_tail - (pos + 1 - start_len) - start_len
                    think_count = tl.maximum(think_count, 0)
                    kmp_s = 0
            else:
                kmp_s = 0  # Simplified: reset on mismatch (no failure table).

        if end_len > 0:
            # KMP advance for </think> end marker.
            expected_e = tl.load(end_ids_ptr + kmp_e)
            if token == expected_e:
                kmp_e = kmp_e + 1
                if kmp_e >= end_len:
                    # Complete </think> match — exiting think mode.
                    in_think = False
                    think_count = 0
                    in_end = False
                    end_count = 0
                    countdown = budget
                    kmp_e = 0
                    kmp_s = 0
            else:
                kmp_e = 0

        if in_think:
            think_count = think_count + 1

    # Update seen_len to current total_len.
    tl.store(seen_len_ptr + req_idx, total_len)

    # --- Budget countdown ---
    if has_budget and budget >= 0:
        # Account for spec draft tokens (not yet committed but about to be sampled).
        spec_count = num_speculative_tokens
        total_predicted = think_count + spec_count
        if in_think and total_predicted > budget:
            # Budget exhausted — transition to end mode.
            in_think = False
            in_end = True
            end_count = 0
            remaining = budget - think_count
            # Force offset: position within spec tokens where forcing starts.
            if remaining > 0 and remaining < spec_count:
                force_off = remaining
            elif remaining <= 0:
                force_off = 0
            else:
                force_off = spec_count  # bonus token position
            tl.store(force_offset_ptr + req_idx, force_off)
            tl.store(force_active_ptr + req_idx, 1)
            tl.store(force_end_count_ptr + req_idx, end_count)
            tl.store(in_think_ptr + req_idx, 0)
            tl.store(in_end_ptr + req_idx, 1)
            tl.store(think_count_ptr + req_idx, 0)
            tl.store(countdown_ptr + req_idx, budget)
            return

        countdown = budget - think_count
        tl.store(countdown_ptr + req_idx, countdown)

    # --- Answer reserve check ---
    if has_reserve and reserve > 0:
        mt = tl.load(max_tokens_ptr + req_idx)
        if mt > 0:
            produced = output_len
            offset = mt - reserve - produced
            if offset <= 0 and in_think:
                # Reserve threshold hit — force end-of-thinking.
                in_think = False
                in_end = True
                end_count = 0
                spec_count = num_speculative_tokens
                tl.store(force_offset_ptr + req_idx, spec_count)
                tl.store(force_active_ptr + req_idx, 1)
                tl.store(force_end_count_ptr + req_idx, 0)
                tl.store(in_think_ptr + req_idx, 0)
                tl.store(in_end_ptr + req_idx, 1)
                tl.store(end_count_ptr + req_idx, 0)
                return

    # --- Multi-token end-marker continuation ---
    if in_end and end_len > 0:
        end_count_val = end_count
        if end_count_val < end_len:
            tl.store(force_active_ptr + req_idx, 1)
            tl.store(force_offset_ptr + req_idx, 0)
            tl.store(force_end_count_ptr + req_idx, end_count_val + 1)
            tl.store(end_count_ptr + req_idx, end_count_val + 1)
            return
        else:
            # Finished multi-token marker — reset.
            in_end = False
            end_count = 0

    # No force this step.
    tl.store(force_active_ptr + req_idx, 0)
    tl.store(in_think_ptr + req_idx, in_think)
    tl.store(in_end_ptr + req_idx, in_end)
    tl.store(think_count_ptr + req_idx, think_count)
    tl.store(end_count_ptr + req_idx, end_count)
    tl.store(kmp_start_ptr + req_idx, kmp_s)
    tl.store(kmp_end_ptr + req_idx, kmp_e)
