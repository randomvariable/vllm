# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.reasoning_monitor import MonitoredObservation
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    MAX_NUM_LOGITS: tl.constexpr,
    SAMPLED_WIDTH: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_logits = end_idx - start_idx
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    offsets = tl.arange(0, MAX_NUM_LOGITS)
    load_mask = (offsets < num_sampled) & (offsets < SAMPLED_WIDTH)
    token_ids = tl.load(
        sampled_ptr + req_idx * sampled_stride + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        flat_sampled_ptr + start_idx + offsets,
        token_ids,
        mask=offsets < num_logits,
    )


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            MAX_NUM_LOGITS=triton.next_power_of_2(sampled.shape[1]),
            SAMPLED_WIDTH=sampled.shape[1],
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[list[MonitoredObservation]] | None,
    ]:
        # Host-side chain-length bound for the ReSET scan. Under adaptive
        # verification the numpy mirror holds the pre-compacted layout, an
        # over-estimate, which the scan masks off per position.
        if self.enable_adaptive_verification:
            max_chain = self.num_speculative_steps + 1
        else:
            max_chain = (
                int(np.diff(cu_num_logits_np).max()) if cu_num_logits_np.size > 1 else 0
            )
        monitoring = (
            not self.enable_adaptive_verification
            and self.sampler.reasoning_monitor.has_enabled_requests
            and np.any(self.sampler.reasoning_monitor.monitoring(idx_mapping_np))
        )
        if monitoring:
            control_logits = self.sampler.apply_sampling_params(
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
                cu_num_logits=cu_num_logits,
                max_chain=max_chain,
                skip_top_k_top_p=True,
            )
            top_k, top_p = self.sampler.sampling_states.get_top_k_top_p(
                expanded_idx_mapping, idx_mapping_np
            )
            # apply_top_k_top_p masks in place; capture_spec below reads the
            # untruncated control stack, so truncation gets its own tensor.
            processed_logits = apply_top_k_top_p(control_logits.clone(), top_k, top_p)
        else:
            control_logits = None
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
                cu_num_logits=cu_num_logits,
                max_chain=max_chain,
            )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        monitor_observations = None
        if monitoring and control_logits is not None:
            monitor_observations = self.sampler.reasoning_monitor.capture_spec(
                logits,
                control_logits,
                sampled,
                num_sampled.cpu().numpy().astype(np.int64),
                pos,
                cu_num_logits_np,
                idx_mapping_np,
            )
        # Commit the ReSET running state for exactly the committed prefix.
        self.sampler.sampling_states.commit_reset_spec(num_sampled)
        return processed_logits, sampled, num_sampled, monitor_observations

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LogprobsTensors | None,
        list[list[MonitoredObservation]] | None,
    ]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []
        monitoring_enabled = (
            self.sampler.reasoning_monitor.has_enabled_requests
            and np.any(
                self.sampler.reasoning_monitor.monitoring(input_batch.idx_mapping_np)
            )
        )
        monitor_observations: list[list[MonitoredObservation]] | None = (
            [[] for _ in range(num_reqs)] if monitoring_enabled else None
        )

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled, chunk_observations = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)
            if monitor_observations is not None and chunk_observations is not None:
                for local_idx, observations in enumerate(chunk_observations):
                    monitor_observations[start + local_idx].extend(observations)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return (
                sampled_chunks[0],
                num_sampled_chunks[0],
                logprobs_tensors,
                monitor_observations,
            )

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors, monitor_observations

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        draft_sampled.masked_fill_(
            input_batch.is_padding[input_batch.logits_indices], -1
        )
        pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors, monitor_observations = (
            self._verify_in_chunks(
                logits,
                input_batch,
                draft_logits,
                draft_sampled,
                pos,
                chunk_logit_limit,
                max_num_logprobs,
            )
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        if monitor_observations is not None:
            committed_counts = num_sampled.cpu().tolist()
            monitor_observations = [
                observations[: int(committed_counts[req_idx])]
                for req_idx, observations in enumerate(monitor_observations)
            ]

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            monitor_observations=monitor_observations,
        )
