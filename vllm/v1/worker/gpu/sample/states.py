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
    def __init__(self, max_num_reqs: int, vocab_size: int, seed: int | None = None):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        # Every TP rank must derive the same fallback request seeds. A private
        # stream avoids rank-local consumers perturbing NumPy's global RNG.
        self._fallback_seed_rng = np.random.default_rng(seed if seed is not None else 0)

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)

        # Reasoning answer-phase temperature. Immutable per-request config; the
        # phase selection happens device side inside the temperature/Gumbel
        # kernels. Per-step entropy temperature (ReSET) is handled separately.
        self.reasoning_answer_temperature = UvaBackedTensor(
            max_num_reqs, dtype=torch.float32
        )
        self.reasoning_answer_temperature_enabled = UvaBackedTensor(
            max_num_reqs, dtype=torch.int32
        )
        # CPU mirror of the answer-phase enable mask, used by the host-side
        # gate that decides whether the kernel path is needed.
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
        # ReSET entropy-threshold temperature (arXiv 2606.13233). Config and
        # running state live in `reset_state`, allocated once the device is
        # known in `bind_reasoning_state`; `use_reset` mirrors the enable mask
        # host-side for the batch gate. MRV2 bypasses logits processors, so
        # ReSET is resolved here from the same on-device core MRV1 uses.
        self.reset_state: Any = None
        self.use_reset = np.zeros(max_num_reqs, dtype=bool)
        self._reset_model_config: Any = None
        self._reset_nl_lut: torch.Tensor | None = None
        self._reset_dnl_lut: torch.Tensor | None = None

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
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = int(
                self._fallback_seed_rng.integers(
                    _NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64
                )
            )
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
        model_config: Any = None,
    ) -> None:
        """Point the schedule at the buffers holding step count and phase.

        When reasoning markers are unavailable the marker cache is replaced by
        constant -1 buffers, which read as "never left the thinking phase" and
        make the answer-temperature override fail closed rather than fire on a
        request whose phase cannot be observed.

        ``model_config`` is retained so the ReSET step-boundary lookup tables
        can be built lazily from the tokenizer on the first ReSET request.
        """
        from vllm.v1.sample.ops.reset import make_reset_state

        self._req_states = req_states
        if cached_last_start is None or cached_last_end is None:
            never = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=device
            )
            cached_last_start = never
            cached_last_end = never
        self._cached_last_start = cached_last_start
        self._cached_last_end = cached_last_end
        self._reset_model_config = model_config
        self.reset_state = make_reset_state(self.max_num_reqs, device)

    def _add_temperature_schedule(
        self, req_idx: int, sampling_params: SamplingParams
    ) -> None:
        answer_temp = sampling_params.reasoning_answer_temperature
        phase = answer_temp is not None
        reset = sampling_params.has_temperature_schedule

        self.use_phase_temperature[req_idx] = phase
        self.use_reset[req_idx] = reset
        self.min_temperature[req_idx] = sampling_params.min_effective_temperature
        self.max_temperature[req_idx] = sampling_params.max_effective_temperature

        self.reasoning_answer_temperature.np[req_idx] = answer_temp if phase else 0.0
        self.reasoning_answer_temperature_enabled.np[req_idx] = phase
        self._schedule_dirty = True

        if self.reset_state is not None:
            self._stage_reset_row(req_idx, sampling_params, reset)

    def _stage_reset_row(
        self, req_idx: int, sampling_params: SamplingParams, reset: bool
    ) -> None:
        """Write ReSET config into the request's row and clear running state."""
        from vllm.v1.sample.ops.reset import T_HIGH, T_LOW, TAU0, W

        st = self.reset_state
        for name in (
            "global_sum",
            "global_n",
            "sw_pos",
            "sw_count",
            "step_sum",
            "step_len",
            "prev_was_nl",
        ):
            getattr(st, name)[req_idx] = 0
        st.sw_ring[req_idx] = 0.0
        st.enabled[req_idx] = 1 if reset else 0
        if not reset:
            return
        p = sampling_params
        st.t_low[req_idx] = (
            p.temperature_low if p.temperature_low is not None else T_LOW
        )
        st.t_high[req_idx] = (
            p.temperature_high if p.temperature_high is not None else T_HIGH
        )
        st.tau0[req_idx] = (
            p.entropy_threshold if p.entropy_threshold is not None else TAU0
        )
        st.window[req_idx] = p.reset_window if p.reset_window is not None else W

    def _ensure_reset_luts(self, device: torch.device) -> None:
        if self._reset_nl_lut is not None:
            return
        from vllm.tokenizers import cached_tokenizer_from_config
        from vllm.v1.sample.ops.reset import get_newline_token_ids

        tokenizer = cached_tokenizer_from_config(self._reset_model_config)
        nl_ids, dnl_ids = get_newline_token_ids(tokenizer)
        nl = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        dnl = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        if nl_ids:
            nl[torch.tensor(nl_ids, device=device)] = True
        if dnl_ids:
            dnl[torch.tensor(dnl_ids, device=device)] = True
        self._reset_nl_lut = nl
        self._reset_dnl_lut = dnl

    def apply_reset(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Apply the ReSET entropy-threshold temperature to ReSET rows.

        Resolves each ReSET row's temperature from the shared on-device core
        and scales those logits rows in place, before the static temperature
        divide (which is a no-op for ReSET rows, whose ``temperature`` is 1.0).
        ReSET requests are admitted only without speculative decoding, so the
        logits rows map one-to-one to requests via ``idx_mapping``.
        """
        if self.reset_state is None:
            return
        mask = self.use_reset[idx_mapping_np]
        if not mask.any():
            return
        from vllm.v1.sample.ops.reset import resolve_reset

        self._ensure_reset_luts(logits.device)
        rows = torch.from_numpy(np.nonzero(mask)[0]).to(logits.device, torch.int64)
        req_idx = idx_mapping[rows].to(torch.int64)
        rs = self._req_states
        prompt_len = rs.prompt_len.gpu[req_idx].to(torch.int64)
        total_len = rs.total_len.gpu[req_idx].to(torch.int64)
        gen_step = total_len - prompt_len
        last_token = rs.last_sampled_tokens[req_idx, 0].to(torch.int64)
        sub = self.reset_state.index_select(req_idx)
        temperature = resolve_reset(
            logits[rows],
            last_token,
            gen_step,
            self._reset_nl_lut,
            self._reset_dnl_lut,
            sub,
        )
        sub.scatter_into(self.reset_state, req_idx)
        logits[rows] = logits[rows] / temperature.unsqueeze(-1)

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()
        if self._schedule_dirty:
            self.reasoning_answer_temperature.copy_to_uva()
            self.reasoning_answer_temperature_enabled.copy_to_uva()
            self._schedule_dirty = False

    def any_dynamic_temperature(self, idx_mapping_np: np.ndarray) -> bool:
        """Whether any request in the batch can change temperature mid-run."""
        return bool(
            np.any(self.use_phase_temperature[idx_mapping_np])
            or np.any(self.use_reset[idx_mapping_np])
        )

    def temperature_schedule(
        self,
        idx_mapping_np: np.ndarray,
    ) -> TemperatureSchedule | None:
        """Answer-phase schedule buffers, or `None` if no request uses one.

        Returning `None` keeps unscheduled batches on the exact static code
        path -- the kernels compile out every schedule branch.
        """
        if not np.any(self.use_phase_temperature[idx_mapping_np]):
            return None
        assert self._cached_last_start is not None
        assert self._cached_last_end is not None
        return TemperatureSchedule(
            answer_temperature=self.reasoning_answer_temperature.gpu,
            answer_enabled=self.reasoning_answer_temperature_enabled.gpu,
            cached_last_start=self._cached_last_start,
            cached_last_end=self._cached_last_end,
        )

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        schedule = self.temperature_schedule(idx_mapping_np)
        if schedule is None:
            temp_np = self.temperature.np[idx_mapping_np]
            if np.all((temp_np == 0.0) | (temp_np == 1.0)):
                # No request requires temperature. Skip the kernel launch.
                return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu, schedule)

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
        return bool(np.any(self.temperature.np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
