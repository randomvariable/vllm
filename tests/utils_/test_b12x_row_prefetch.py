# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Next-step span prediction and staging for the disk row prefetcher."""

import queue
import threading

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.common.engram_prefetch import compress_engram_tokens
from vllm.utils.b12x_row_prefetch import (
    PrefetchTarget,
    RowPrefetcher,
    _Job,
    _run_job,
    plan_next_step_spans,
)

# (a) decoding, (b) final prefill chunk, (c)/(d) still prefilling.
IDX_MAPPING = np.array([7, 3, 11, 5], dtype=np.intp)
SCHEDULED = np.array([4, 4, 6, 8], dtype=np.int32)
COMPUTED_PREFILL = np.array([10, 8, 0, 16], dtype=np.int32)
PREFILL_LEN = np.array([10, 12, 20, 40], dtype=np.int32)
IS_PREFILLING = np.array([False, True, True, True], dtype=np.bool_)


def _plan(max_prefill_tokens):
    return plan_next_step_spans(
        idx_mapping_np=IDX_MAPPING,
        num_scheduled_tokens=SCHEDULED,
        num_computed_prefill_tokens=COMPUTED_PREFILL,
        prefill_len=PREFILL_LEN,
        is_prefilling=IS_PREFILLING,
        max_prefill_tokens=max_prefill_tokens,
    )


def test_finished_prefill_decodes_and_next_chunk_is_scheduled_length():
    """A chunk that ends the prompt is not lookahead; a mid-prompt chunk is."""
    spans = _plan(2048)
    assert spans.decode_positions.tolist() == [0, 1]
    assert spans.prefill_starts.tolist() == [6, 24]
    assert spans.prefill_lens.tolist() == [6, 8]
    assert spans.prefill_rows.tolist() == IDX_MAPPING[[2, 3]].tolist()


def test_prefill_lookahead_is_capped_by_the_budget():
    """The crossing span is truncated and later spans are dropped."""
    assert _plan(10).prefill_lens.tolist() == [6, 4]
    assert _plan(5).prefill_lens.tolist() == [5]
    assert _plan(5).prefill_rows.tolist() == [IDX_MAPPING[2]]


HISTORY_LEN = 3
HISTORY_PAD = 7
# Request rows 5 and 9 decode; row 11 keeps prefilling from position 6.
BATCH_ROWS = np.array([5, 9, 11], dtype=np.intp)
TOKENS = torch.zeros(16, 64, dtype=torch.int32)
for _row in (5, 9, 11):
    TOKENS[_row] = torch.arange(64, dtype=torch.int32) % 900 + 1
NUM_COMPUTED = torch.zeros(16, dtype=torch.int32)
NUM_COMPUTED[5] = 10
NUM_COMPUTED[9] = 2
LAST_SAMPLED = torch.zeros(16, 1, dtype=torch.int64)
LAST_SAMPLED[5] = 21
LAST_SAMPLED[9] = 3
DRAFTS = torch.tensor([[30, 31], [32, 33], [0, 0]], dtype=torch.int64)


class _Buffer:
    def __init__(self, values):
        self.gpu = values


def _batch(num_reqs=3):
    return type(
        "InputBatch",
        (),
        {
            "num_reqs": num_reqs,
            "idx_mapping_np": BATCH_ROWS,
            "num_scheduled_tokens": np.ones(num_reqs, dtype=np.int32),
            "num_computed_prefill_tokens_np": np.array([10, 10, 2], dtype=np.int32),
            "prefill_len_np": np.array([10, 10, 20], dtype=np.int32),
            "is_prefilling_np": np.array([False, False, True], dtype=np.bool_),
        },
    )()


def _req_states():
    return type(
        "RequestState",
        (),
        {
            "all_token_ids": _Buffer(TOKENS),
            "num_computed_tokens": _Buffer(NUM_COMPUTED),
            "last_sampled_tokens": LAST_SAMPLED,
        },
    )()


def _prefetcher(targets=()):
    """A prefetcher without device allocations: the gathers are device-agnostic."""
    prefetcher = object.__new__(RowPrefetcher)
    prefetcher._targets = tuple(targets)
    prefetcher._history_len = HISTORY_LEN
    prefetcher._history_pad = HISTORY_PAD
    prefetcher._device = torch.device("cpu")
    prefetcher._max_query_width = 3
    prefetcher._slots = [torch.zeros(4096, dtype=torch.int64) for _ in range(2)]
    prefetcher._free_slots = queue.Queue()
    prefetcher._jobs = queue.Queue()
    prefetcher._disabled = threading.Event()
    return prefetcher


class _Event:
    def synchronize(self):
        return None


class _Table:
    """Stand-in for a b12x ``DiskTable``."""

    def __init__(self, result=True):
        self.result = result
        self.seen = []

    def prefetch(self, ids):
        self.seen.append(ids)
        return self.result


class _NoPrefetch:
    """A DiskTable from a b12x build that has no ``prefetch``."""


def _job(prefetcher, slot, num_decode_rows, query_width, prefill_lens, targets):
    return _Job(
        slot_index=0,
        slot=slot,
        event=_Event(),
        num_decode_rows=num_decode_rows,
        query_width=query_width,
        prefill_lens=np.asarray(prefill_lens, dtype=np.int64),
        history_len=HISTORY_LEN,
        targets=tuple(targets),
        free_slots=prefetcher._free_slots,
        disabled=prefetcher._disabled,
    )


def _staged_spans():
    """Two decode rows (5, 9) and one prefill span on row 11 from position 6."""
    return plan_next_step_spans(
        idx_mapping_np=BATCH_ROWS,
        num_scheduled_tokens=np.array([1, 1, 4], dtype=np.int32),
        num_computed_prefill_tokens=np.array([10, 10, 2], dtype=np.int32),
        prefill_len=np.array([10, 10, 20], dtype=np.int32),
        is_prefilling=np.array([False, False, True], dtype=np.bool_),
        max_prefill_tokens=2048,
    )


def test_decode_window_is_history_then_sampled_then_drafts():
    """Negative history positions pad, and the staged draft chain extends the row."""
    decode = _prefetcher()._decode_tokens(
        _batch(), _req_states(), _staged_spans(), DRAFTS
    )
    assert decode.shape == (2, HISTORY_LEN + 3)
    # Row 5 has positions 7,8,9 committed; row 9 has only two, so the oldest pads.
    assert decode[0].tolist() == [8, 9, 10, 21, 30, 31]
    assert decode[1].tolist() == [HISTORY_PAD, 1, 2, 3, 32, 33]


def test_prefill_window_carries_history_before_the_next_chunk():
    """The hashed window needs the preceding context, not just the new chunk."""
    spans = _staged_spans()
    assert spans.prefill_starts.tolist() == [6]
    assert spans.prefill_lens.tolist() == [4]
    prefill = _prefetcher()._prefill_tokens(_req_states(), spans)
    # History positions 3,4,5 followed by the chunk at 6..9 of request row 11.
    assert prefill.tolist() == [4, 5, 6, 7, 8, 9, 10]


def test_worker_prefetches_the_rows_of_both_staged_blocks():
    """Slot bytes round-trip into exactly the per-block rows the target predicts."""
    spans = _staged_spans()
    prefetcher = _prefetcher()
    decode = prefetcher._decode_tokens(_batch(), _req_states(), spans, DRAFTS)
    prefill = prefetcher._prefill_tokens(_req_states(), spans)
    slot = prefetcher._slots[0]
    slot[: decode.numel()].copy_(decode.reshape(-1))
    slot[decode.numel() : decode.numel() + prefill.numel()].copy_(prefill)
    table = _Table()
    # One id per block row, taken from its newest token: the last draft of each
    # decode row and the last token of the chunk. A wrongly sliced block moves
    # these.
    target = PrefetchTarget(
        resolve_table=lambda: table, row_ids=lambda block: block[:, -1]
    )
    _run_job(_job(prefetcher, slot, 2, 3, spans.prefill_lens, [target]))
    assert [ids.tolist() for ids in table.seen] == [[10, 31, 33]]


@pytest.mark.parametrize(
    "table", [_NoPrefetch(), _Table(result=False)], ids=["no-prefetch", "cache-off"]
)
def test_worker_disables_after_an_unusable_prefetch(table):
    """A table that cannot prefetch must stop the prefetcher, not raise."""
    prefetcher = _prefetcher()
    target = PrefetchTarget(
        resolve_table=lambda: table, row_ids=lambda block: block.reshape(-1)
    )
    _run_job(
        _job(
            prefetcher,
            torch.zeros(4096, dtype=torch.int64),
            num_decode_rows=1,
            query_width=3,
            prefill_lens=[],
            targets=[target],
        )
    )
    assert prefetcher.disabled


def test_compress_engram_tokens_marks_dead_tokens():
    """Out-of-vocabulary, negative and image-sentinel ids hash as the pad row."""
    from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

    token_map = torch.arange(0, 130_000 * 7, 7, dtype=torch.int64)
    raw = torch.tensor(
        [0, 4, 129_263, IMAGE_SENTINEL_BASE_ID, 130_000, -1], dtype=torch.int64
    )
    assert compress_engram_tokens(raw, token_map).tolist() == [
        0,
        28,
        129_263 * 7,
        -1,
        -1,
        -1,
    ]
