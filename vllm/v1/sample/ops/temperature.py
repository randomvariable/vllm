# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side reasoning answer-phase temperature resolution.

The resolved temperature is never computed on the host. Callers stage the
static per-request temperature plus a reasoning-phase flag into persistent GPU
buffers, and this module turns those into a per-row temperature with plain
tensor ops -- so the work stays on device, is Inductor-fusable, and runs
unchanged on CPU.

This covers only the reasoning answer-phase override (a request may switch to a
different temperature once it leaves its thinking block). Per-step entropy
temperature control is provided separately by ReSET
(`vllm.v1.sample.ops.reset`), which is a distinct policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

_SAMPLING_EPS = 1e-5


@dataclass(eq=False)
class TemperatureSchedule:
    """Persistent per-request buffers backing the answer-phase temperature.

    Every tensor is allocated once at `max_num_reqs` and updated in place, so
    the schedule is safe to read from inside a CUDA graph.

    Attributes:
        base: Static per-request temperature (the value used outside the
            answer phase).
        answer_temperature: Temperature to use during the answer phase.
        answer_enabled: Non-zero where the answer-phase override applies.
        reasoning_phase: Non-zero while a request is inside its thinking
            block. The answer-phase override fires only where this is zero
            *and* `entered_reasoning` is set.
        entered_reasoning: Latched non-zero once a request has been observed
            inside a thinking block, so a request that never reasoned is never
            treated as being in the answer phase.
    """

    base: torch.Tensor
    answer_temperature: torch.Tensor
    answer_enabled: torch.Tensor
    reasoning_phase: torch.Tensor
    entered_reasoning: torch.Tensor
    reset_active: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemperatureSchedule):
            return NotImplemented
        if self.reset_active != other.reset_active:
            return False
        return all(
            torch.equal(getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
            if f.name != "reset_active"
        )

    def narrow(self, num_reqs: int) -> TemperatureSchedule:
        """View of the first `num_reqs` rows, without copying."""
        return TemperatureSchedule(
            base=self.base[:num_reqs],
            answer_temperature=self.answer_temperature[:num_reqs],
            answer_enabled=self.answer_enabled[:num_reqs],
            reasoning_phase=self.reasoning_phase[:num_reqs],
            entered_reasoning=self.entered_reasoning[:num_reqs],
            reset_active=self.reset_active,
        )

    def index_select(self, rows: torch.Tensor) -> TemperatureSchedule:
        """Gather one schedule row per entry of `rows`."""
        return TemperatureSchedule(
            base=self.base[rows],
            answer_temperature=self.answer_temperature[rows],
            answer_enabled=self.answer_enabled[rows],
            reasoning_phase=self.reasoning_phase[rows],
            entered_reasoning=self.entered_reasoning[rows],
            reset_active=self.reset_active,
        )


def resolve_temperature(schedule: TemperatureSchedule) -> torch.Tensor:
    """Resolve the effective temperature for each row, on device.

    Args:
        schedule: Row-aligned schedule buffers; one entry per output row.

    Returns:
        A float tensor of per-row temperatures. Rows outside the answer phase
        return `base` unchanged, bit for bit.
    """
    # The override applies only after a request has been inside a thinking
    # block and has since left it. A request still thinking, or one that never
    # entered reasoning at all, keeps its base value.
    in_answer = (schedule.entered_reasoning != 0) & (schedule.reasoning_phase == 0)
    return torch.where(
        (schedule.answer_enabled != 0) & in_answer,
        schedule.answer_temperature,
        schedule.base,
    )


def divide_by_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    all_random: bool,
) -> torch.Tensor:
    """Divide `logits` in place by an already-resolved per-row temperature.

    The resolved temperature is left untouched so the caller can still use a
    zero to mean greedy; only the divisor substitutes a placeholder.

    Args:
        logits: `[num_rows, vocab]` logits, modified in place.
        temperature: `[num_rows]` resolved temperatures.
        all_random: Whether every row is known to sample randomly, which lets
            the divide-by-zero guard be skipped.

    Returns:
        The same tensor, divided in place.
    """
    divisor = temperature
    if not all_random:
        divisor = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)
    return logits.div_(divisor.unsqueeze(-1))
