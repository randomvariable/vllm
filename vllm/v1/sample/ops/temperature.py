# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side step-aware temperature resolution.

The resolved temperature is never computed on the host. Callers stage
immutable per-request configuration plus a step counter and a reasoning-phase
flag into persistent GPU buffers, and this module turns those into a per-row
temperature with plain tensor ops -- so the work stays on device, is
Inductor-fusable, and runs unchanged on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

_SAMPLING_EPS = 1e-5


@dataclass(eq=False)
class TemperatureSchedule:
    """Persistent per-request buffers backing the step-aware temperature.

    Every tensor is allocated once at `max_num_reqs` and updated in place, so
    the schedule is safe to read from inside a CUDA graph.

    Attributes:
        base: Static temperature, i.e. the value at step 0.
        final: Temperature reached once `anneal_steps` have been generated.
        anneal_steps: Length of the anneal ramp, in generated tokens.
        schedule_enabled: Non-zero where the anneal ramp applies.
        answer_temperature: Temperature to use during the answer phase.
        answer_enabled: Non-zero where the answer-phase override applies.
        generated_steps: Tokens generated so far, per request.
        reasoning_phase: Non-zero while a request is inside its thinking
            block. The answer-phase override fires only where this is zero
            *and* `entered_reasoning` is set.
        entered_reasoning: Latched non-zero once a request has been observed
            inside a thinking block, so a request that never reasoned is never
            treated as being in the answer phase.
    """

    base: torch.Tensor
    final: torch.Tensor
    anneal_steps: torch.Tensor
    schedule_enabled: torch.Tensor
    answer_temperature: torch.Tensor
    answer_enabled: torch.Tensor
    generated_steps: torch.Tensor
    reasoning_phase: torch.Tensor
    entered_reasoning: torch.Tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemperatureSchedule):
            return NotImplemented
        return all(
            torch.equal(getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
        )

    def narrow(self, num_reqs: int) -> TemperatureSchedule:
        """View of the first `num_reqs` rows, without copying."""
        return TemperatureSchedule(
            base=self.base[:num_reqs],
            final=self.final[:num_reqs],
            anneal_steps=self.anneal_steps[:num_reqs],
            schedule_enabled=self.schedule_enabled[:num_reqs],
            answer_temperature=self.answer_temperature[:num_reqs],
            answer_enabled=self.answer_enabled[:num_reqs],
            generated_steps=self.generated_steps[:num_reqs],
            reasoning_phase=self.reasoning_phase[:num_reqs],
            entered_reasoning=self.entered_reasoning[:num_reqs],
        )

    def index_select(self, rows: torch.Tensor) -> TemperatureSchedule:
        """Gather one schedule row per entry of `rows`."""
        return TemperatureSchedule(
            base=self.base[rows],
            final=self.final[rows],
            anneal_steps=self.anneal_steps[rows],
            schedule_enabled=self.schedule_enabled[rows],
            answer_temperature=self.answer_temperature[rows],
            answer_enabled=self.answer_enabled[rows],
            generated_steps=self.generated_steps[rows],
            reasoning_phase=self.reasoning_phase[rows],
            entered_reasoning=self.entered_reasoning[rows],
        )


def resolve_temperature(
    schedule: TemperatureSchedule,
    step_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve the effective temperature for each row, on device.

    Args:
        schedule: Row-aligned schedule buffers; one entry per output row.
        step_offset: Extra steps to add to `generated_steps` for each row, used
            where a request contributes several rows (speculative draft
            positions). `None` means every row is at the request's own step.

    Returns:
        A float32 tensor of per-row temperatures. Rows whose request has no
        schedule return `base` unchanged, bit for bit.
    """
    base = schedule.base
    step = schedule.generated_steps
    if step_offset is not None:
        step = step + step_offset

    anneal_steps = schedule.anneal_steps.clamp_min(1).to(base.dtype)
    progress = (step.to(base.dtype) / anneal_steps).clamp_(0.0, 1.0)
    annealed = torch.lerp(base, schedule.final, progress)
    temperature = torch.where(schedule.schedule_enabled != 0, annealed, base)

    # The override applies only after a request has been inside a thinking
    # block and has since left it. A request still thinking, or one that never
    # entered reasoning at all, keeps its base or annealed value.
    in_answer = (schedule.entered_reasoning != 0) & (schedule.reasoning_phase == 0)
    return torch.where(
        (schedule.answer_enabled != 0) & in_answer,
        schedule.answer_temperature,
        temperature,
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
