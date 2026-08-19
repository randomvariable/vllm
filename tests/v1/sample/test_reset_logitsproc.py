# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-lifecycle tests for the ReSET logits processor.

The temperature policy itself is proven bit-exact against the reference oracle
in ``test_reset_policy.py``; here we check the *plumbing* the logits processor
adds on top -- staging config on add, resolving the batch in ``apply``, and
relocating per-request running state across ``move``/``swap``/``remove`` -- by
comparing against ``resolve_reset`` driven directly on an isolated single row.
"""

from __future__ import annotations

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.logits_processor.reset import ReSETLogitsProcessor
from vllm.v1.sample.ops.reset import (
    T_HIGH,
    T_LOW,
    TAU0,
    ResetState,
    W,
    resolve_reset,
)


@pytest.fixture(autouse=True)
def _pin_cpu_default_device():
    """Keep these CPU tests order-independent.

    `tests/v1/sample/test_sampler.py` calls `torch.set_default_device(...)`
    without restoring it, so any later test that allocates a tensor inherits
    that device. These tests compare against a CPU oracle and must not.
    """
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


_VOCAB = 96
_NL, _DNL = 5, 7


class _StubModelConfig:
    def get_vocab_size(self) -> int:
        return _VOCAB


class _StubSchedulerConfig:
    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs


class _StubVllmConfig:
    def __init__(self, max_num_seqs: int) -> None:
        self.model_config = _StubModelConfig()
        self.scheduler_config = _StubSchedulerConfig(max_num_seqs)


def _make_processor(max_num_seqs: int) -> ReSETLogitsProcessor:
    proc = ReSETLogitsProcessor(
        _StubVllmConfig(max_num_seqs), torch.device("cpu"), False
    )
    # Inject the step-boundary luts so `_ensure_luts` skips the tokenizer.
    nl = torch.zeros(_VOCAB, dtype=torch.bool)
    dnl = torch.zeros(_VOCAB, dtype=torch.bool)
    nl[_NL] = True
    dnl[_DNL] = True
    proc._nl_lut = nl
    proc._dnl_lut = dnl
    return proc


def _reset_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        temperature_low=T_LOW,
        temperature_high=T_HIGH,
        entropy_threshold=TAU0,
        reset_window=W,
    )


def _single_row_state() -> ResetState:
    def z(dt):
        return torch.zeros(1, dtype=dt)

    return ResetState(
        enabled=torch.ones(1, dtype=torch.int32),
        base=torch.ones(1, dtype=torch.float32),
        t_low=torch.full((1,), T_LOW),
        t_high=torch.full((1,), T_HIGH),
        tau0=torch.full((1,), TAU0),
        window=torch.full((1,), W, dtype=torch.int32),
        global_sum=z(torch.float32),
        global_n=z(torch.int64),
        sw_ring=torch.zeros(1, W, dtype=torch.float32),
        sw_pos=z(torch.int64),
        sw_count=z(torch.int64),
        step_sum=z(torch.float32),
        step_len=z(torch.int64),
        prev_was_nl=z(torch.int32),
    )


def _stream(length, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(length, _VOCAB, generator=g) * 2.0
    tokens = torch.randint(0, _VOCAB, (length,), generator=g).tolist()
    for i in range(5, length, 13):
        tokens[i] = _DNL
    for i in range(8, length, 19):
        tokens[i] = _NL
        if i + 1 < length:
            tokens[i + 1] = _NL
    return logits, tokens


def _applied_temp(raw_row, scaled_row):
    """Recover the uniform scale applied to a logits row."""
    k = int(torch.argmax(raw_row.abs()))
    return raw_row[k].item() / scaled_row[k].item()


def test_processor_matches_direct_core_single_request():
    length = 80
    logits, tokens = _stream(length, seed=3)
    proc = _make_processor(4)
    ref_state = _single_row_state()

    output_ids: list[int] = []
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, _reset_params(), None, output_ids)],
            moved=[],
        )
    )

    for t in range(length):
        last = torch.tensor([tokens[t - 1] if t >= 1 else 0], dtype=torch.int64)
        step = torch.tensor([t], dtype=torch.int64)
        expected = resolve_reset(
            logits[t : t + 1], last, step, proc._nl_lut, proc._dnl_lut, ref_state
        ).item()

        row = logits[t : t + 1].clone()
        proc.apply(row)
        applied = _applied_temp(logits[t], row[0])
        assert abs(applied - expected) < 1e-4, f"step {t}: {applied} vs {expected}"

        output_ids.append(tokens[t])


def test_state_follows_swap():
    length = 60
    la, ta = _stream(length, seed=10)
    lb, tb = _stream(length, seed=20)
    proc = _make_processor(4)
    ref_a = _single_row_state()
    ref_b = _single_row_state()

    out_a: list[int] = []
    out_b: list[int] = []
    proc.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[
                (0, _reset_params(), None, out_a),
                (1, _reset_params(), None, out_b),
            ],
            moved=[],
        )
    )

    swap_at = 25
    for t in range(length):
        if t == swap_at:
            proc.update_state(
                BatchUpdate(
                    batch_size=2,
                    removed=[],
                    added=[],
                    moved=[(0, 1, MoveDirectionality.SWAP)],
                )
            )
        # Row 0 always carries stream A, row 1 stream B -- after the swap the
        # processor must have relocated A's running state to row 1 and B's to 0.
        pos_a = 1 if t >= swap_at else 0
        pos_b = 0 if t >= swap_at else 1

        batch = torch.empty(2, _VOCAB)
        batch[pos_a] = la[t]
        batch[pos_b] = lb[t]

        last = torch.tensor(
            [ta[t - 1] if t >= 1 else 0, tb[t - 1] if t >= 1 else 0], dtype=torch.int64
        )
        step = torch.tensor([t, t], dtype=torch.int64)
        exp_a = resolve_reset(
            la[t : t + 1], last[:1], step[:1], proc._nl_lut, proc._dnl_lut, ref_a
        ).item()
        exp_b = resolve_reset(
            lb[t : t + 1], last[1:], step[1:], proc._nl_lut, proc._dnl_lut, ref_b
        ).item()

        raw = batch.clone()
        proc.apply(batch)
        got_a = _applied_temp(raw[pos_a], batch[pos_a])
        got_b = _applied_temp(raw[pos_b], batch[pos_b])
        assert abs(got_a - exp_a) < 1e-4, f"A step {t}: {got_a} vs {exp_a}"
        assert abs(got_b - exp_b) < 1e-4, f"B step {t}: {got_b} vs {exp_b}"

        out_a.append(ta[t])
        out_b.append(tb[t])


def test_removed_request_frees_state():
    proc = _make_processor(4)
    out: list[int] = []
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, _reset_params(), None, out)],
            moved=[],
        )
    )
    assert 0 in proc.output_ids
    proc.update_state(BatchUpdate(batch_size=0, removed=[0], added=[], moved=[]))
    assert 0 not in proc.output_ids
    assert proc.state.enabled[0].item() == 0
    assert proc.state.global_n[0].item() == 0
