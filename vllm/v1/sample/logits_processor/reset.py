# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched ReSET temperature-scaling logits processor.

ReSET (arXiv 2606.13233; github.com/aiha-lab/ReSET) sets each token's decoding
temperature from token- and step-level entropy. The reference is a per-request
processor that reads the entropy back to the host with ``.item()`` every token;
this batched version keeps the whole policy on device via
`vllm.v1.sample.ops.reset.resolve_reset`, so no per-token device-to-host sync is
needed.

The processor owns one row of running state per request, kept aligned with the
persistent batch through `update_state`. On `apply` it resolves the temperature
for every ReSET row in a single batched call and scales those rows in place;
rows without ReSET are untouched.

Enable it per request by setting any ReSET knob on ``SamplingParams``
(``temperature_low``, ``temperature_high``, ``entropy_threshold``,
``reset_window``) with ``temperature=1.0`` so the temperature is applied once,
by this processor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm.v1.sample.ops.reset import (
    T_HIGH,
    T_LOW,
    TAU0,
    ResetState,
    W,
    get_newline_token_ids,
    make_reset_state,
    resolve_reset,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ReSETLogitsProcessor(LogitsProcessor):
    """Applies the ReSET entropy-threshold temperature policy, batched."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.model_config = vllm_config.model_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_reqs = max_num_reqs

        # Full-batch running state, indexed by persistent batch position.
        self.state = make_reset_state(max_num_reqs, device)
        # Live references to each ReSET row's output token ids, for boundary
        # detection and the step counter. Absent index -> not a ReSET request.
        self.output_ids: dict[int, list[int]] = {}

        # Step-boundary lookup tables are built lazily on the first ReSET
        # request, so models/requests that never use ReSET pay nothing.
        self._nl_lut: torch.Tensor | None = None
        self._dnl_lut: torch.Tensor | None = None

    def is_argmax_invariant(self) -> bool:
        return False

    def _ensure_luts(self) -> None:
        if self._nl_lut is not None:
            return
        from vllm.tokenizers import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(self.model_config)
        vocab = self.model_config.get_vocab_size()
        nl_ids, dnl_ids = get_newline_token_ids(tokenizer)
        nl_lut = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        dnl_lut = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        if nl_ids:
            nl_lut[torch.tensor(nl_ids, device=self.device)] = True
        if dnl_ids:
            dnl_lut[torch.tensor(dnl_ids, device=self.device)] = True
        self._nl_lut = nl_lut
        self._dnl_lut = dnl_lut

    def _clear_row(self, index: int) -> None:
        for name in _STATE_FIELDS:
            getattr(self.state, name)[index] = 0
        self.state.base[index] = 1.0
        self.state.t_low[index] = T_LOW
        self.state.t_high[index] = T_HIGH
        self.state.tau0[index] = TAU0
        self.state.window[index] = W

    def _stage_row(self, index: int, params: SamplingParams) -> None:
        self._clear_row(index)
        self.state.enabled[index] = 1
        self.state.t_low[index] = (
            params.temperature_low if params.temperature_low is not None else T_LOW
        )
        self.state.t_high[index] = (
            params.temperature_high if params.temperature_high is not None else T_HIGH
        )
        self.state.tau0[index] = (
            params.entropy_threshold if params.entropy_threshold is not None else TAU0
        )
        self.state.window[index] = (
            params.reset_window if params.reset_window is not None else W
        )

    def _move_row(self, dst: int, src: int) -> None:
        for name in _ALL_FIELDS:
            getattr(self.state, name)[dst] = getattr(self.state, name)[src].clone()

    def _swap_rows(self, a: int, b: int) -> None:
        for name in _ALL_FIELDS:
            buf = getattr(self.state, name)
            tmp = buf[a].clone()
            buf[a] = buf[b].clone()
            buf[b] = tmp

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for index in batch_update.removed:
            if self.output_ids.pop(index, None) is not None:
                self._clear_row(index)

        for index, params, _prompt_ids, output_ids in batch_update.added:
            if params.has_temperature_schedule:
                self._ensure_luts()
                self._stage_row(index, params)
                self.output_ids[index] = output_ids
            elif self.output_ids.pop(index, None) is not None:
                # A non-ReSET request reused a slot a ReSET request held.
                self._clear_row(index)

        for adx, bdx, direct in batch_update.moved:
            a_on = adx in self.output_ids
            b_on = bdx in self.output_ids
            if not (a_on or b_on):
                continue
            if direct == MoveDirectionality.SWAP:
                self._swap_rows(adx, bdx)
                a_ids = self.output_ids.get(adx)
                b_ids = self.output_ids.get(bdx)
                if b_ids is not None:
                    self.output_ids[adx] = b_ids
                else:
                    self.output_ids.pop(adx, None)
                if a_ids is not None:
                    self.output_ids[bdx] = a_ids
                else:
                    self.output_ids.pop(bdx, None)
            else:
                self._move_row(bdx, adx)
                if a_on:
                    self.output_ids[bdx] = self.output_ids.pop(adx)
                    self._clear_row(adx)
                else:
                    self.output_ids.pop(bdx, None)
                    self._clear_row(bdx)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.output_ids:
            return logits
        num_reqs = logits.shape[0]
        rows = [i for i in self.output_ids if i < num_reqs]
        if not rows:
            return logits
        rows.sort()

        last_ids = []
        gen_steps = []
        for i in rows:
            out = self.output_ids[i]
            last_ids.append(out[-1] if out else 0)
            gen_steps.append(len(out))
        row_idx = torch.tensor(rows, dtype=torch.int64, device=self.device)
        last_token = torch.tensor(last_ids, dtype=torch.int64, device=self.device)
        gen_step = torch.tensor(gen_steps, dtype=torch.int64, device=self.device)

        assert self._nl_lut is not None and self._dnl_lut is not None
        sub_state = self.state.index_select(row_idx)
        temperature = resolve_reset(
            logits[row_idx],
            last_token,
            gen_step,
            self._nl_lut,
            self._dnl_lut,
            sub_state,
        )
        sub_state.scatter_into(self.state, row_idx)
        logits[row_idx] = logits[row_idx] / temperature.unsqueeze(-1)
        return logits


# Field groups for row bookkeeping.
_STATE_FIELDS = (
    "enabled",
    "global_sum",
    "global_n",
    "sw_ring",
    "sw_pos",
    "sw_count",
    "step_sum",
    "step_len",
    "prev_was_nl",
)
_ALL_FIELDS = tuple(f.name for f in ResetState.__dataclass_fields__.values())
