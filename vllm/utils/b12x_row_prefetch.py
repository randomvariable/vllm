# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advisory host-side prefetch of disk-backed PLE/Engram rows.

A disk-backed row table reads its rows through a host row cache. Waiting for
the device hash to name the rows costs a full disk round trip inside the
forward pass, so the runner predicts the tokens the next step will hash and
hands their row ids to the table's ``prefetch`` from a worker thread. The
engine thread only records non-blocking device-to-host copies and a CUDA event;
all hashing, de-duplication and queueing happens off it.

Prediction is advisory. The synchronous read path stays the correctness path,
so a wrong guess costs a cache miss and nothing else.
"""

from __future__ import annotations

import queue
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

# TODO: fixed per-step bound on prefill lookahead tokens. b12x prefetch()
# normalizes ids with a per-id Python loop; this caps GIL time taken from the
# engine thread.
PREFILL_PREFETCH_MAX_TOKENS = 2048

_STOP = object()


@dataclass(frozen=True)
class PrefetchTarget:
    """One disk table and the row-id function that predicts its next-step ids.

    ``row_ids`` maps an ``(R, H + Q)`` int64 CPU tensor of raw token ids, the
    ``H`` committed history columns followed by the ``Q`` query columns, to the
    flat global row ids those tokens hash to.
    """

    resolve_table: Callable[[], Any | None]
    row_ids: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class NextStepSpans:
    """The token windows the next step is expected to hash.

    ``decode_positions`` are the batch positions that decode next step; the
    others are still prefilling, described by request-state row, first token
    position and length of their next chunk.
    """

    decode_positions: np.ndarray
    prefill_rows: np.ndarray
    prefill_starts: np.ndarray
    prefill_lens: np.ndarray


def plan_next_step_spans(
    *,
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_computed_prefill_tokens: np.ndarray,
    prefill_len: np.ndarray,
    is_prefilling: np.ndarray,
    max_prefill_tokens: int,
) -> NextStepSpans:
    """Split this step's batch into next-step decodes and prefill lookahead.

    Every input is a length-``num_reqs`` array in batch order at the start of
    this step. A request still prefilling keeps taking tokens next step, so its
    next chunk is predicted as this step's scheduled length, capped by the
    tokens left in the prompt and by ``max_prefill_tokens`` across the batch.
    """
    num_reqs = int(idx_mapping_np.shape[0])
    scheduled = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    computed = np.asarray(num_computed_prefill_tokens[:num_reqs], dtype=np.int64)
    total = np.asarray(prefill_len[:num_reqs], dtype=np.int64)
    prefilling = np.asarray(is_prefilling[:num_reqs], dtype=bool)

    next_start = computed + scheduled
    still_prefill = prefilling & (next_start < total)
    positions = np.flatnonzero(still_prefill)

    rows: list[int] = []
    starts: list[int] = []
    lens: list[int] = []
    budget = int(max_prefill_tokens)
    for position in positions.tolist():
        length = min(
            int(total[position] - next_start[position]), int(scheduled[position])
        )
        if length <= 0:
            continue
        length = min(length, budget)
        rows.append(position)
        starts.append(int(next_start[position]))
        lens.append(length)
        budget -= length
        if budget <= 0:
            break
    return NextStepSpans(
        decode_positions=np.flatnonzero(~still_prefill).astype(np.int64),
        prefill_rows=idx_mapping_np[rows].astype(np.int64),
        prefill_starts=np.asarray(starts, dtype=np.int64),
        prefill_lens=np.asarray(lens, dtype=np.int64),
    )


@dataclass(frozen=True)
class _Job:
    slot_index: int
    slot: torch.Tensor
    event: torch.cuda.Event
    num_decode_rows: int
    query_width: int
    prefill_lens: np.ndarray
    history_len: int
    targets: tuple[PrefetchTarget, ...]
    free_slots: queue.Queue[int]
    disabled: threading.Event


def _prefetch_worker(jobs: queue.Queue[Any]) -> None:
    """Hash staged tokens and hand the rows to each disk table, off the engine."""
    while True:
        job = jobs.get()
        if job is _STOP:
            return
        try:
            _run_job(job)
        except Exception as exc:
            logger.warning_once(
                "Disabling PLE/Engram row prefetch after error: %r", exc
            )
            job.disabled.set()
        finally:
            job.free_slots.put(job.slot_index)


def _run_job(job: _Job) -> None:
    job.event.synchronize()
    history_len = job.history_len
    blocks = []
    if job.num_decode_rows:
        count = job.num_decode_rows * (history_len + job.query_width)
        blocks.append(job.slot[:count].view(job.num_decode_rows, -1))
    offset = sum(block.numel() for block in blocks)
    for length in job.prefill_lens.tolist():
        count = history_len + int(length)
        blocks.append(job.slot[offset : offset + count].view(1, count))
        offset += count
    if not blocks:
        return
    for target in job.targets:
        table = target.resolve_table()
        if table is None:
            continue
        if not callable(getattr(table, "prefetch", None)):
            logger.warning_once(
                "VLLM_PLE_PREFETCH=1 requires a b12x build whose DiskTable "
                "provides prefetch(ids) (local-inference-lab/flashinfer#44). "
                "Disabling VLLM_PLE_PREFETCH."
            )
            job.disabled.set()
            return
        ids = torch.unique(torch.cat([target.row_ids(block) for block in blocks]))
        if table.prefetch(ids) is False:
            logger.warning_once(
                "DiskRowCache.prefetch is inactive; set a "
                "B12X_ROW_CACHE_ROWS/B12X_ROW_CACHE_BYTES size (GDS tables "
                "cannot prefetch). Disabling VLLM_PLE_PREFETCH."
            )
            job.disabled.set()
            return


class RowPrefetcher:
    """Stage the next step's tokens and prefetch their rows for every target."""

    def __init__(
        self,
        targets: tuple[PrefetchTarget, ...],
        *,
        history_len: int,
        history_pad: int,
        max_num_reqs: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        self._targets = tuple(targets)
        self._history_len = int(history_len)
        self._history_pad = int(history_pad)
        self._device = device
        self._max_query_width = 1 + int(num_speculative_tokens)
        capacity = (
            max_num_reqs * (self._history_len + self._max_query_width)
            + PREFILL_PREFETCH_MAX_TOKENS
            + max_num_reqs * self._history_len
        )
        self._slots = [
            torch.empty(capacity, dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event() for _ in self._slots]
        self._free_slots: queue.Queue[int] = queue.Queue()
        for index in range(len(self._slots)):
            self._free_slots.put(index)
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._disabled = threading.Event()
        threading.Thread(
            target=_prefetch_worker,
            args=(self._jobs,),
            name="vllm-row-prefetch",
            daemon=True,
        ).start()
        weakref.finalize(self, self._jobs.put, _STOP)

    @property
    def disabled(self) -> bool:
        return self._disabled.is_set()

    def submit(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        draft_tokens: torch.Tensor | None,
    ) -> None:
        """Record the next step's tokens and queue the prefetch work.

        Never blocks on the GPU: the only device interaction is asynchronous
        copies into a free pinned slot plus one event record. A saturated slot
        ring drops the request rather than waiting for a slot, and any failure
        disables the prefetcher so the engine thread pays no further cost.
        """
        if self._disabled.is_set() or input_batch.num_reqs == 0:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        try:
            slot_index = self._free_slots.get_nowait()
        except queue.Empty:
            return
        try:
            job = self._stage(slot_index, input_batch, req_states, draft_tokens)
        except Exception as exc:
            self._free_slots.put(slot_index)
            self._disabled.set()
            logger.warning_once(
                "Disabling PLE/Engram row prefetch after error: %r", exc
            )
            return
        if job is not None:
            self._jobs.put_nowait(job)

    def _stage(
        self,
        slot_index: int,
        input_batch: InputBatch,
        req_states: RequestState,
        draft_tokens: torch.Tensor | None,
    ) -> _Job | None:
        num_reqs = input_batch.num_reqs
        spans = plan_next_step_spans(
            idx_mapping_np=input_batch.idx_mapping_np[:num_reqs],
            num_scheduled_tokens=input_batch.num_scheduled_tokens[:num_reqs],
            num_computed_prefill_tokens=input_batch.num_computed_prefill_tokens_np[
                :num_reqs
            ],
            prefill_len=input_batch.prefill_len_np[:num_reqs],
            is_prefilling=input_batch.is_prefilling_np[:num_reqs],
            max_prefill_tokens=PREFILL_PREFETCH_MAX_TOKENS,
        )
        decode = self._decode_tokens(input_batch, req_states, spans, draft_tokens)
        prefill = self._prefill_tokens(req_states, spans)
        parts = [block.reshape(-1) for block in (decode, prefill) if block is not None]
        if not parts:
            self._free_slots.put(slot_index)
            return None
        query_width = self._max_query_width
        if decode is not None:
            query_width = int(decode.shape[1]) - self._history_len
        slot = self._slots[slot_index]
        flat = torch.cat(parts)
        slot[: flat.numel()].copy_(flat, non_blocking=True)
        event = self._events[slot_index]
        event.record(torch.cuda.current_stream())
        return _Job(
            slot_index=slot_index,
            slot=slot,
            event=event,
            num_decode_rows=int(spans.decode_positions.shape[0]),
            query_width=query_width,
            prefill_lens=spans.prefill_lens.copy(),
            history_len=self._history_len,
            targets=self._targets,
            free_slots=self._free_slots,
            disabled=self._disabled,
        )

    def _decode_tokens(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        spans: NextStepSpans,
        draft_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Committed history, last sampled token and staged drafts per decode row."""
        positions = spans.decode_positions
        if positions.shape[0] == 0:
            return None
        history_len = self._history_len
        rows = torch.as_tensor(
            input_batch.idx_mapping_np[positions],
            dtype=torch.int64,
            device=self._device,
        )
        base = req_states.num_computed_tokens.gpu[rows].long()[:, None]
        cols = base + torch.arange(
            -history_len, 0, dtype=torch.int64, device=self._device
        )
        history = torch.where(
            cols >= 0,
            req_states.all_token_ids.gpu[rows[:, None], cols.clamp_min(0)].long(),
            self._history_pad,
        )
        query = req_states.last_sampled_tokens[rows]
        if (
            draft_tokens is not None
            and draft_tokens.dim() == 2
            and draft_tokens.shape[1] > 0
        ):
            batch_pos = torch.as_tensor(
                positions, dtype=torch.int64, device=self._device
            )
            query = torch.cat((query, draft_tokens[batch_pos].long()), dim=1)
        return torch.cat((history, query), dim=1)

    def _prefill_tokens(
        self, req_states: RequestState, spans: NextStepSpans
    ) -> torch.Tensor | None:
        """History plus next-chunk tokens of every still-prefilling request, flat."""
        if spans.prefill_rows.shape[0] == 0:
            return None
        history_len = self._history_len
        rows, cols = [], []
        for row, start, length in zip(
            spans.prefill_rows.tolist(),
            spans.prefill_starts.tolist(),
            spans.prefill_lens.tolist(),
        ):
            rows.append(np.full(history_len + length, row, dtype=np.int64))
            cols.append(np.arange(start - history_len, start + length, dtype=np.int64))
        row_ids = torch.as_tensor(
            np.concatenate(rows), dtype=torch.int64, device=self._device
        )
        token_pos = torch.as_tensor(
            np.concatenate(cols), dtype=torch.int64, device=self._device
        )
        gathered = req_states.all_token_ids.gpu[row_ids, token_pos.clamp_min(0)].long()
        return torch.where(token_pos >= 0, gathered, self._history_pad)
