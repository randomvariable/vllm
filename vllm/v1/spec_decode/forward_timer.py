# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import torch

from vllm.v1.spec_decode.timing import SpecForwardTimings


@dataclasses.dataclass
class _Slot:
    """One step's event pairs plus the batch shape they were recorded under."""

    target_start: torch.cuda.Event
    target_end: torch.cuda.Event
    draft_start: torch.cuda.Event
    draft_end: torch.cuda.Event
    num_tokens: int = 0
    num_reqs: int = 0
    num_spec_tokens: int = 0
    has_target: bool = False
    has_draft: bool = False


class SpecForwardTimer:
    """Deferred CUDA event timing for target and drafter forwards.

    Wall-clock timing is meaningless here: kernel launches and CUDA graph
    replays return immediately, so timing around the launch measures launch
    overhead rather than GPU work. CUDA events measure the device work.

    Events are read two steps after they were recorded, from a three-slot ring.
    The recording step never calls `elapsed_time`, and a slot whose events have
    not completed is dropped rather than waited on. This is the same discipline
    as the async D2H read in
    `vllm/v1/worker/gpu/spec_decode/dspark/confidence_scheduler.py:438`
    ("not done yet -- do NOT stall the hot path"). Do not "simplify" this back
    into a synchronous read.

    Args:
        num_slots: Ring depth. Three gives a two-step deferral, which is the
            minimum that keeps the read off the recording step under async
            scheduling.
    """

    def __init__(self, num_slots: int = 3) -> None:
        if num_slots < 3:
            raise ValueError("num_slots must be at least 3.")
        self._slots = [
            _Slot(
                target_start=torch.cuda.Event(enable_timing=True),
                target_end=torch.cuda.Event(enable_timing=True),
                draft_start=torch.cuda.Event(enable_timing=True),
                draft_end=torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_slots)
        ]
        self._num_slots = num_slots
        self._step = 0

    def start_step(self, num_tokens: int, num_reqs: int, num_spec_tokens: int) -> None:
        """Open a new ring slot for this step.

        Args:
            num_tokens: Tokens in the target forward.
            num_reqs: Requests in the batch, used as the load bucket key.
            num_spec_tokens: Draft length scheduled for this step.
        """
        self._step += 1
        slot = self._slots[self._step % self._num_slots]
        slot.num_tokens = num_tokens
        slot.num_reqs = num_reqs
        slot.num_spec_tokens = num_spec_tokens
        slot.has_target = False
        slot.has_draft = False

    @property
    def _current(self) -> _Slot:
        return self._slots[self._step % self._num_slots]

    def record_target_start(self) -> None:
        self._current.target_start.record()

    def record_target_end(self) -> None:
        slot = self._current
        slot.target_end.record()
        slot.has_target = True

    def record_draft_start(self) -> None:
        self._current.draft_start.record()

    def record_draft_end(self) -> None:
        """Close the drafter window.

        Called even when `K == 0`: the proposer still runs a cache-sync forward
        before returning an empty tensor, and that cost is exactly what makes
        `K == 0` non-free. Skipping it would teach a governor that `K == 0`
        costs nothing.
        """
        slot = self._current
        slot.draft_end.record()
        slot.has_draft = True

    def take_ready(self) -> SpecForwardTimings | None:
        """Read the slot recorded two steps ago, if its events have completed.

        Returns:
            Timings for that step, or `None` when the ring is not yet warm or
            the events are still outstanding. Never blocks.
        """
        slot = self._slots[(self._step + 1) % self._num_slots]
        if not slot.has_target:
            return None
        last = slot.draft_end if slot.has_draft else slot.target_end
        if not last.query():
            return None
        draft_ms = None
        if slot.has_draft:
            draft_ms = slot.draft_start.elapsed_time(slot.draft_end)
        return SpecForwardTimings(
            target_ms=slot.target_start.elapsed_time(slot.target_end),
            draft_ms=draft_ms,
            num_tokens=slot.num_tokens,
            num_reqs=slot.num_reqs,
            num_spec_tokens=slot.num_spec_tokens,
        )
