# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitor-only reasoning telemetry capture (MGT-B style, Phase B).

Captures per-committed-token observables at the V2 sampler for the layered
reasoning-control monitor: full-vocabulary entropy before and after the
control stack, the chosen-token log-probability after commit, and the accepted
token id + committed position.

The observables are all reused from tensors the sampler already computes, so
capture adds no extra forward pass:

* pre-control entropy: ``reset_entropy(raw_logits)`` (ReSET already gates this
  computation when a request uses the entropy-threshold temperature);
* post-control entropy: ``reset_entropy(processed_logits)`` after
  ``apply_sampling_params``;
* chosen-token logprob: column 0 of ``compute_topk_scores`` over the same
  processed logits.

Under speculative decoding only *committed* tokens advance the monitor state;
draft-only and rejected rows are dropped with the rest of the chain. The
committed signal is the per-request ``num_sampled`` returned by the rejection
sampler, applied here by trimming each request's captured row history to
``num_sampled`` entries before they leave the worker.

This module is strictly opt in and strictly monitor only: a request that does
not enable the monitor is byte-for-byte behaviourally inert and takes no new
kernel work. The recurrence state (windows, CUSUM, refractory, alarms,
accounting) lives on the scheduler-owned ``Request`` object, not here; this
module only produces the attributed observation vector for committed positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.reset import reset_entropy


@dataclass
class MonitoredObservation:
    """One committed-position observation attributed to a request.

    ``entropy_pre`` and ``entropy_post`` are the full-vocabulary Shannon
    entropies (nats) before and after the control stack; ``logprob`` is the
    chosen token's log-probability after it was committed; ``token_id`` and
    ``position`` are the accepted token and its committed position.

    The pre/post split makes the marker penalty's entropy contribution (and
    therefore its coupling with ReSET) attributable, which is the point of
    recording both.
    """

    entropy_pre: float
    entropy_post: float
    logprob: float
    token_id: int
    position: int


class ReasoningMonitor:
    """V2 sampler-side capture of per-committed-token observables."""

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        max_commit_tokens: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.max_commit_tokens = max_commit_tokens
        self._monitor_enabled = np.zeros(max_num_reqs, dtype=bool)
        self._enabled_count = 0

    @property
    def has_enabled_requests(self) -> bool:
        """Return whether any live request opted into monitoring."""
        return self._enabled_count > 0

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        enabled = bool(sampling_params.reasoning_monitor)
        previous = bool(self._monitor_enabled[req_idx])
        self._monitor_enabled[req_idx] = enabled
        self._enabled_count += int(enabled) - int(previous)

    def apply_staged_writes(self) -> None:
        """Keep sampler state API-compatible; monitoring is host-side only."""
        return

    def monitoring(self, idx_mapping_np: np.ndarray) -> np.ndarray:
        """Host-side gate: which rows in a batch have monitoring enabled."""
        return self._monitor_enabled[idx_mapping_np]

    def capture(
        self,
        raw_logits: torch.Tensor,
        processed_logits: torch.Tensor | None,
        sampled_token_ids: torch.Tensor,
        committed_counts: np.ndarray,
        commit_offsets: np.ndarray,
        request_indices: np.ndarray | None = None,
    ) -> list[list[MonitoredObservation]]:
        """Compute committed-position observations for monitored requests.

        Args:
            raw_logits: ``[num_requests, vocab]`` logits before any sampling
                processors ran (the exact tensor the control stack started
                from).
            processed_logits: ``[num_requests, vocab]`` after all sampling
                processors (penalties, marker penalty, ReSET, temperature) ran.
                ``None`` when no request requested logprobs in this step (the
                chosen logprob then cannot be attributed without an extra
                reduction, so it is left at ``nan``).
            sampled_token_ids: ``[num_requests]`` chosen token ids.
            committed_counts: ``[num_requests]`` number of committed tokens
                this step (spec decode: the accepted prefix length).
            commit_offsets: ``[num_requests]`` position within the request of
                the first committed token this step (the base for positions).

        Returns:
            One list per request, containing ``committed_counts[i]``
            observations for request ``i`` (empty when it is not being
            monitored). Monitored requests reuse ReSET entropy where present;
            entropy is computed with the same ``reset_entropy`` primitive the
            ReSET policy uses.
        """
        num_reqs = sampled_token_ids.shape[0]
        results: list[list[MonitoredObservation]] = [[] for _ in range(num_reqs)]
        if request_indices is None:
            request_indices = np.arange(num_reqs, dtype=np.int64)
        if (
            raw_logits.shape[0] != num_reqs
            or committed_counts.shape[0] != num_reqs
            or commit_offsets.shape[0] != num_reqs
            or request_indices.shape[0] != num_reqs
            or np.any(committed_counts > 1)
        ):
            return results

        rows = np.flatnonzero(self._monitor_enabled[request_indices])
        if rows.size == 0:
            return results

        entropy_pre = reset_entropy(raw_logits)
        entropy_post = (
            reset_entropy(processed_logits) if processed_logits is not None else None
        )
        chosen_probs = None
        if processed_logits is not None:
            from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

            chosen_probs = compute_token_logprobs(
                processed_logits, sampled_token_ids[:, None]
            )[:, 0]

        for i in rows:
            if committed_counts[i] != 1:
                continue
            results[i].append(
                MonitoredObservation(
                    entropy_pre=float(entropy_pre[i].item()),
                    entropy_post=(
                        float(entropy_post[i].item())
                        if entropy_post is not None
                        else float("nan")
                    ),
                    logprob=(
                        float(chosen_probs[i].item())
                        if chosen_probs is not None
                        else float("nan")
                    ),
                    token_id=int(sampled_token_ids[i].item()),
                    position=int(commit_offsets[i]),
                )
            )
        return results

    def capture_spec(
        self,
        raw_logits: torch.Tensor,
        control_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        committed_counts: np.ndarray,
        positions: torch.Tensor,
        cu_num_logits: np.ndarray,
        request_indices: np.ndarray,
    ) -> list[list[MonitoredObservation]]:
        """Capture only accepted target rows from a speculative chain."""
        num_reqs = sampled_token_ids.shape[0]
        results: list[list[MonitoredObservation]] = [[] for _ in range(num_reqs)]
        if (
            sampled_token_ids.ndim != 2
            or raw_logits.shape != control_logits.shape
            or raw_logits.shape[0] != positions.shape[0]
            or cu_num_logits.shape[0] != num_reqs + 1
            or request_indices.shape[0] != num_reqs
            or committed_counts.shape[0] != num_reqs
        ):
            return results
        monitored = self._monitor_enabled[request_indices]
        if not np.any(monitored):
            return results

        row_indices: list[int] = []
        token_ids: list[int] = []
        request_rows: list[list[tuple[int, int]]] = [[] for _ in range(num_reqs)]
        for req_idx in np.flatnonzero(monitored):
            start = int(cu_num_logits[req_idx])
            end = int(cu_num_logits[req_idx + 1])
            count = min(max(int(committed_counts[req_idx]), 0), end - start)
            for offset in range(count):
                row = start + offset
                token_id = int(sampled_token_ids[req_idx, offset].item())
                request_rows[req_idx].append((row, token_id))
                row_indices.append(row)
                token_ids.append(token_id)
        if not row_indices:
            return results

        rows_tensor = torch.tensor(
            row_indices, device=control_logits.device, dtype=torch.long
        )
        tokens_tensor = torch.tensor(
            token_ids, device=control_logits.device, dtype=torch.long
        )
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

        entropy_pre = reset_entropy(raw_logits).detach().cpu().numpy()
        entropy_post = reset_entropy(control_logits).detach().cpu().numpy()
        chosen_probs = (
            compute_token_logprobs(
                control_logits.index_select(0, rows_tensor), tokens_tensor[:, None]
            )[:, 0]
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        positions_np = positions.detach().cpu().numpy()
        for req_idx, req_rows in enumerate(request_rows):
            for local_idx, (row, token_id) in enumerate(req_rows):
                flat_idx = sum(len(rows) for rows in request_rows[:req_idx]) + local_idx
                results[req_idx].append(
                    MonitoredObservation(
                        entropy_pre=float(entropy_pre[row]),
                        entropy_post=float(entropy_post[row]),
                        logprob=float(chosen_probs[flat_idx]),
                        token_id=token_id,
                        position=int(positions_np[row]),
                    )
                )
        return results
