# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import TemperatureSchedule, apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


class SamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)

        # Step-aware temperature. These are immutable per-request config; the
        # per-row step, interpolation and phase selection all happen device
        # side inside the temperature/Gumbel kernels.
        self.temperature_final = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.temperature_anneal_steps = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.temperature_schedule_enabled = UvaBackedTensor(
            max_num_reqs, dtype=torch.int32
        )
        self.reasoning_answer_temperature = UvaBackedTensor(
            max_num_reqs, dtype=torch.float32
        )
        self.reasoning_answer_temperature_enabled = UvaBackedTensor(
            max_num_reqs, dtype=torch.int32
        )
        # CPU mirrors of the two enable masks, used by the host-side gates that
        # decide whether the kernel and the device-native sampler are needed.
        self.use_temperature_schedule = np.zeros(max_num_reqs, dtype=bool)
        self.use_phase_temperature = np.zeros(max_num_reqs, dtype=bool)
        # Widest interval a request's temperature can span, so a statically
        # non-greedy request that can anneal to zero still routes correctly.
        self.min_temperature = np.zeros(max_num_reqs, dtype=np.float32)
        self.max_temperature = np.zeros(max_num_reqs, dtype=np.float32)
        self._schedule_dirty = False
        # Set by `bind_reasoning_state`; the step count and reasoning phase are
        # read from buffers these own.
        self._req_states: Any = None
        self._cached_last_start: torch.Tensor | None = None
        self._cached_last_end: torch.Tensor | None = None
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.temperature.np[req_idx] = sampling_params.temperature
        self._add_temperature_schedule(req_idx, sampling_params)
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

    def bind_reasoning_state(
        self,
        req_states: Any,
        cached_last_start: torch.Tensor | None,
        cached_last_end: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        """Point the schedule at the buffers holding step count and phase.

        When reasoning markers are unavailable the marker cache is replaced by
        constant -1 buffers, which read as "never left the thinking phase" and
        make the answer-temperature override fail closed rather than fire on a
        request whose phase cannot be observed.
        """
        self._req_states = req_states
        if cached_last_start is None or cached_last_end is None:
            never = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=device
            )
            cached_last_start = never
            cached_last_end = never
        self._cached_last_start = cached_last_start
        self._cached_last_end = cached_last_end

    def _add_temperature_schedule(
        self, req_idx: int, sampling_params: SamplingParams
    ) -> None:
        final = sampling_params.temperature_final
        anneal_steps = sampling_params.temperature_anneal_steps
        schedule = final is not None and anneal_steps is not None
        answer_temp = sampling_params.reasoning_answer_temperature
        phase = answer_temp is not None

        self.use_temperature_schedule[req_idx] = schedule
        self.use_phase_temperature[req_idx] = phase
        self.min_temperature[req_idx] = sampling_params.min_effective_temperature
        self.max_temperature[req_idx] = sampling_params.max_effective_temperature

        self.temperature_final.np[req_idx] = final if schedule else 0.0
        self.temperature_anneal_steps.np[req_idx] = anneal_steps if schedule else 0
        self.temperature_schedule_enabled.np[req_idx] = schedule
        self.reasoning_answer_temperature.np[req_idx] = answer_temp if phase else 0.0
        self.reasoning_answer_temperature_enabled.np[req_idx] = phase
        self._schedule_dirty = True

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()
        if self._schedule_dirty:
            self.temperature_final.copy_to_uva()
            self.temperature_anneal_steps.copy_to_uva()
            self.temperature_schedule_enabled.copy_to_uva()
            self.reasoning_answer_temperature.copy_to_uva()
            self.reasoning_answer_temperature_enabled.copy_to_uva()
            self._schedule_dirty = False

    def any_dynamic_temperature(self, idx_mapping_np: np.ndarray) -> bool:
        """Whether any request in the batch can change temperature mid-run."""
        return bool(
            np.any(self.use_temperature_schedule[idx_mapping_np])
            or np.any(self.use_phase_temperature[idx_mapping_np])
        )

    def temperature_schedule(
        self,
        idx_mapping_np: np.ndarray,
        expanded_local_pos: torch.Tensor,
    ) -> TemperatureSchedule | None:
        """Schedule buffers for this batch, or `None` if no request uses one.

        Returning `None` keeps unscheduled batches on the exact static code
        path -- the kernels compile out every schedule branch.
        """
        if not self.any_dynamic_temperature(idx_mapping_np):
            return None
        assert self._req_states is not None, (
            "bind_reasoning_state() must run before a temperature schedule"
        )
        assert self._cached_last_start is not None
        assert self._cached_last_end is not None
        return TemperatureSchedule(
            temperature_final=self.temperature_final.gpu,
            anneal_steps=self.temperature_anneal_steps.gpu,
            schedule_enabled=self.temperature_schedule_enabled.gpu,
            answer_temperature=self.reasoning_answer_temperature.gpu,
            answer_enabled=self.reasoning_answer_temperature_enabled.gpu,
            expanded_local_pos=expanded_local_pos,
            total_len=self._req_states.total_len.gpu,
            prompt_len=self._req_states.prompt_len.gpu,
            cached_last_start=self._cached_last_start,
            cached_last_end=self._cached_last_end,
        )

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        schedule = self.temperature_schedule(idx_mapping_np, expanded_local_pos)
        if schedule is None:
            temp_np = self.temperature.np[idx_mapping_np]
            if np.all((temp_np == 0.0) | (temp_np == 1.0)):
                # No request requires temperature. Skip the kernel launch.
                return

        apply_temperature(
            logits, expanded_idx_mapping, self.temperature.gpu, schedule=schedule
        )

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p)

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        # A scheduled request can reach zero on a later step, so gate on the
        # smallest temperature it could ever resolve to rather than the value
        # it happens to start at.
        return bool(np.any(self.min_temperature[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
