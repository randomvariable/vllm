# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReSET policy correctness against the reference implementation.

The oracle ``_ReferenceReSET`` is ``ReSETRequest`` from the paper's reference
implementation (github.com/aiha-lab/ReSET, reset/reset/logits_processor.py),
copied verbatim except that it returns the resolved temperature instead of the
scaled logits so the batched core can be compared directly. The batched core
under test is ``vllm.v1.sample.ops.reset.resolve_reset``.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.sample.ops.reset import (
    T_HIGH,
    T_LOW,
    TAU0,
    ResetState,
    W,
    reset_entropy,
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


class _ReferenceReSET:
    """Verbatim reference ``ReSETRequest`` (returns temperature, not logits)."""

    def __init__(self, t_high, t_low, tau_raw, window, nl_ids, dnl_ids):
        self.t_high = float(t_high)
        self.t_low = float(t_low)
        self.tau_raw = float(tau_raw)
        self.window = int(window)
        self.nl_ids = frozenset(int(x) for x in nl_ids)
        self.dnl_ids = frozenset(int(x) for x in dnl_ids)
        self.sw_buffer: list[float] = []
        self.step_buffer: list[float] = []
        self._global_sum = 0.0
        self._global_n = 0

    def _is_boundary(self, output_ids) -> bool:
        n = len(output_ids)
        if n == 0:
            return False
        if output_ids[-1] in self.dnl_ids:
            return True
        return (
            n >= 2 and output_ids[-1] in self.nl_ids and output_ids[-2] in self.nl_ids
        )

    def temperature(self, output_ids, logits):
        """Return ``(temp, ent, H_step, gmean)`` — internals for tie checks."""
        if self._is_boundary(output_ids):
            self.step_buffer = []
        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).item())
        if len(self.step_buffer) < self.window:
            H_step = (
                sum(self.sw_buffer) / len(self.sw_buffer) if self.sw_buffer else ent
            )
        else:
            H_step = sum(self.step_buffer) / len(self.step_buffer)
        gmean = self._global_sum / self._global_n if self._global_n > 0 else H_step
        high_step = H_step > gmean
        if not high_step:
            temp = self.t_high if ent >= self.tau_raw else self.t_low
        else:
            temp = self.t_high if ent >= H_step else self.t_low
        self.step_buffer.append(ent)
        self.sw_buffer.append(ent)
        if len(self.sw_buffer) > self.window:
            del self.sw_buffer[0]
        self._global_sum += ent
        self._global_n += 1
        return temp, ent, H_step, gmean


def _luts(vocab, nl_ids, dnl_ids, device):
    nl = torch.zeros(vocab, dtype=torch.bool, device=device)
    dnl = torch.zeros(vocab, dtype=torch.bool, device=device)
    nl[list(nl_ids)] = True
    dnl[list(dnl_ids)] = True
    return nl, dnl


def _fresh_state(num_rows, device, window=W, t_low=T_LOW, t_high=T_HIGH, tau0=TAU0):
    def full(v, dt):
        return torch.full((num_rows,), v, dtype=dt, device=device)

    def zeros(dt):
        return torch.zeros(num_rows, dtype=dt, device=device)

    return ResetState(
        enabled=full(1, torch.int32),
        base=full(1.0, torch.float32),
        t_low=full(t_low, torch.float32),
        t_high=full(t_high, torch.float32),
        tau0=full(tau0, torch.float32),
        window=full(window, torch.int32),
        global_sum=zeros(torch.float32),
        global_n=zeros(torch.int64),
        sw_ring=torch.zeros(num_rows, window, dtype=torch.float32, device=device),
        sw_pos=zeros(torch.int64),
        sw_count=zeros(torch.int64),
        step_sum=zeros(torch.float32),
        step_len=zeros(torch.int64),
        prev_was_nl=zeros(torch.int32),
    )


def _random_stream(length, vocab, nl_ids, dnl_ids, seed):
    """A token/logits stream with newline tokens injected to force boundaries."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(length, vocab, generator=g) * 2.0
    tokens = torch.randint(0, vocab, (length,), generator=g).tolist()
    # Inject single- and double-newline boundaries at assorted positions.
    for i in range(5, length, 17):
        tokens[i] = next(iter(dnl_ids))
    for i in range(9, length, 23):
        tokens[i] = next(iter(nl_ids))
        if i + 1 < length:
            tokens[i + 1] = next(iter(nl_ids))
    return logits, tokens


def test_reset_entropy_matches_manual():
    vocab = 128
    logits = torch.randn(4, vocab)
    probs = torch.softmax(logits, dim=-1)
    expected = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1)
    assert torch.allclose(reset_entropy(logits), expected, atol=1e-6)
    # A near-uniform row is near log(vocab); a peaked row is near zero.
    uniform = torch.zeros(1, vocab)
    peaked = torch.full((1, vocab), -30.0)
    peaked[0, 0] = 30.0
    import math

    assert abs(reset_entropy(uniform).item() - math.log(vocab)) < 1e-4
    assert reset_entropy(peaked).item() < 1e-3


def _agrees(core_temp, ref) -> bool:
    """Whether the core and reference resolved the same temperature decision.

    The two possible temperatures (``T_low``, ``T_high``) are far apart, so a
    ``1e-4`` tolerance cleanly identifies the decision while absorbing the
    float32-vs-float64 gap in representing a value like ``0.1``. A genuine flip
    is allowed only when the reference sat on a decision boundary (a float tie
    at the threshold or between the step estimate and the global mean).
    """
    ref_temp, ent, h_step, gmean = ref
    if abs(core_temp - ref_temp) < 1e-4:
        return True
    eps = 1e-3
    return (
        abs(ent - TAU0) <= eps or abs(ent - h_step) <= eps or abs(h_step - gmean) <= eps
    )


def test_batched_core_matches_reference_single_stream():
    vocab, length = 96, 120
    nl_ids, dnl_ids = {5}, {7}
    device = torch.device("cpu")
    nl_lut, dnl_lut = _luts(vocab, nl_ids, dnl_ids, device)
    logits, tokens = _random_stream(length, vocab, nl_ids, dnl_ids, seed=1234)

    engine = _ReferenceReSET(T_HIGH, T_LOW, TAU0, W, nl_ids, dnl_ids)
    state = _fresh_state(1, device)

    for t in range(length):
        row = logits[t : t + 1]
        ref = engine.temperature(tokens[:t], logits[t])
        last = torch.tensor([tokens[t - 1] if t >= 1 else 0], dtype=torch.int64)
        step = torch.tensor([t], dtype=torch.int64)
        core_temp = resolve_reset(row, last, step, nl_lut, dnl_lut, state).item()
        assert _agrees(core_temp, ref), f"step {t}: core {core_temp} vs ref {ref}"


def test_batched_core_matches_reference_multi_row():
    vocab, length, rows = 96, 90, 5
    nl_ids, dnl_ids = {3}, {8}
    device = torch.device("cpu")
    nl_lut, dnl_lut = _luts(vocab, nl_ids, dnl_ids, device)

    streams = [
        _random_stream(length, vocab, nl_ids, dnl_ids, seed=100 + r)
        for r in range(rows)
    ]
    engines = [
        _ReferenceReSET(T_HIGH, T_LOW, TAU0, W, nl_ids, dnl_ids) for _ in range(rows)
    ]
    state = _fresh_state(rows, device)

    for t in range(length):
        row_logits = torch.stack([streams[r][0][t] for r in range(rows)])
        last = torch.tensor(
            [streams[r][1][t - 1] if t >= 1 else 0 for r in range(rows)],
            dtype=torch.int64,
        )
        step = torch.full((rows,), t, dtype=torch.int64)
        core = resolve_reset(row_logits, last, step, nl_lut, dnl_lut, state)
        for r in range(rows):
            ref = engines[r].temperature(streams[r][1][:t], streams[r][0][t])
            assert _agrees(core[r].item(), ref), (
                f"row {r} step {t}: {core[r].item()} vs {ref}"
            )


def test_step_segmentation_changes_output():
    """Boundary detection is load-bearing: removing it changes temperatures."""
    vocab, length = 96, 100
    nl_ids, dnl_ids = {5}, {7}
    device = torch.device("cpu")
    nl_lut, dnl_lut = _luts(vocab, nl_ids, dnl_ids, device)
    empty = torch.zeros(vocab, dtype=torch.bool, device=device)
    logits, tokens = _random_stream(length, vocab, nl_ids, dnl_ids, seed=77)

    def run(nl, dnl):
        state = _fresh_state(1, device)
        out = []
        for t in range(length):
            last = torch.tensor([tokens[t - 1] if t >= 1 else 0], dtype=torch.int64)
            step = torch.tensor([t], dtype=torch.int64)
            out.append(
                resolve_reset(logits[t : t + 1], last, step, nl, dnl, state).item()
            )
        return out

    segmented = run(nl_lut, dnl_lut)
    unsegmented = run(empty, empty)
    assert segmented != unsegmented
