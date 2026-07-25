# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions derived from ROCm/ATOM (https://github.com/ROCm/ATOM),
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc., licensed under MIT.
"""DSpark confidence-scheduled verification: Hardware-Aware Prefix Scheduler.

Port of ATOM's DSpark confidence scheduler (paper §3.2.2, Algorithm 1): given
per-position confidence scores for a batch of draft blocks and an engine
throughput profile SPS(B), choose a per-request verification length ell_r that
maximizes system-wide token throughput Theta = tau * SPS(B).

This is GATED by VLLM_DSPARK_CONFIDENCE_SCHEDULING (default OFF). When disabled,
behavior is identical to Phase-1 (static verify length = num_speculative_steps).

Pure PyTorch, 100% gfx1151-portable (no CDNA/FP8/MLA dependencies).
"""

from __future__ import annotations

from typing import Sequence

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_confidence_scheduling_enabled() -> bool:
    """Return True if DSpark confidence scheduling is enabled via env var."""
    return envs.VLLM_DSPARK_CONFIDENCE_SCHEDULING


# Early-stop in the greedy admission loop. Required for losslessness under the
# smooth-SPS assumption (paper §3.5). Disabling it does an unconstrained global
# search that crosses the SPS sawtooth (paper §5.2) but then needs the async
# two-step causal barrier to stay lossless — that variant lives at the call site.
_GREEDY_EARLY_STOP = True


def survival_probabilities(confidence: torch.Tensor) -> torch.Tensor:
    """Cumulative-product survival probabilities a_{r,j} = prod_{i<=j} c_{r,i}.

    Args:
        confidence: [R, gamma] per-position conditional acceptance probs in (0,1).
    Returns:
        [R, gamma] cumulative survival probabilities, monotonically non-increasing
        along the block axis.
    """
    if confidence.ndim != 2:
        raise ValueError(
            f"confidence must be [R, gamma], got {tuple(confidence.shape)}"
        )
    return confidence.float().cumprod(dim=1)


def calibrate_confidence(
    confidence: torch.Tensor,
    sts_temperatures: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Sequential Temperature Scaling (STS) to raw confidence (paper §3.2.1).

    STS is an order-preserving per-position temperature on the confidence LOGIT:
    c_k <- sigmoid(logit(c_k) / T_k). Temperatures are fit offline (held-out
    grid search minimizing cumulative-product ECE); this is the cheap apply side.

    Args:
        confidence: [R, gamma] raw sigmoid confidence in (0, 1).
        sts_temperatures: [gamma] positive per-position temperatures, or None
            (T=1, i.e. no calibration — functionally valid, just less accurate).
    Returns:
        [R, gamma] calibrated confidence in (0, 1).
    """
    c = confidence.float().clamp(1e-6, 1 - 1e-6)
    if sts_temperatures is None:
        return c
    t = sts_temperatures.float().to(c.device)
    if t.numel() != c.shape[1]:
        raise ValueError(
            f"sts_temperatures length {t.numel()} != block size {c.shape[1]}"
        )
    if torch.any(t <= 0):
        raise ValueError(
            "STS temperatures must be strictly positive (order-preserving)."
        )
    logit = torch.log(c) - torch.log1p(-c)  # logit(c)
    return torch.sigmoid(logit / t)


def build_sps_table(
    token_points: Sequence[int],
    sps_points: Sequence[float],
    max_b: int,
) -> torch.Tensor:
    """Build a dense sps_table[B] (B in 0..max_b) from measured sample points.

    The engine throughput SPS(B) (steps/sec at a forward of B tokens) is only
    profiled at a handful of batch sizes (the captured CUDA-graph sizes). This
    densifies those samples into a per-B lookup the scheduler can index in O(1).

    Interpolation: piecewise-linear between measured points, flat-held outside the
    measured range. Linear keeps the curve smooth (the scheduler's early-stop
    losslessness assumes a smoothly decaying SPS; paper §3.5).

    Args:
        token_points: measured forward sizes (num_tokens), any order.
        sps_points: throughput (steps/sec) at each token_point. Same length.
        max_b: table covers indices 0..max_b inclusive.
    Returns:
        [max_b + 1] fp32 tensor; sps_table[B] = SPS at a B-token forward.
    """
    if len(token_points) != len(sps_points):
        raise ValueError("token_points and sps_points must have equal length")
    if len(token_points) == 0:
        raise ValueError("need at least one measured point")
    if max_b < 0:
        raise ValueError("max_b must be non-negative")

    pts = sorted(zip(token_points, sps_points))  # by token count ascending
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]

    table = torch.empty(max_b + 1, dtype=torch.float32)
    j = 0
    for b in range(max_b + 1):
        if b <= xs[0]:
            table[b] = ys[0]
        elif b >= xs[-1]:
            table[b] = ys[-1]
        else:
            # Advance to the segment [xs[j], xs[j+1]] containing b.
            while j + 1 < len(xs) and b > xs[j + 1]:
                j += 1
            x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
            w = (b - x0) / (x1 - x0) if x1 > x0 else 0.0
            table[b] = y0 + w * (y1 - y0)
    return table


def schedule_prefix_lengths_tensor(
    confidence: torch.Tensor,
    sps_table: torch.Tensor,
    *,
    sts_temperatures: torch.Tensor | None = None,
    early_stop: bool = _GREEDY_EARLY_STOP,
) -> torch.Tensor:
    """Vectorized Hardware-Aware Prefix Scheduler — returns ell as a device
    tensor with NO host sync (no .item()/.tolist()), for the decode hot path.

    Vectorization insight: survival a_{r,j} is monotone non-increasing in j
    (cumprod), so admitting the GLOBAL top-m tokens by survival automatically
    respects intra-block prefix order. The greedy admission therefore collapses
    to a single descending sort + cumulative sum:

        admit top-m (m=0..N) -> B(m) = R + m ; tau(m) = R + cumsum(sorted a)[m]
        Theta(m) = tau(m) * SPS(B(m))

    Early-stop returns the first local max of Theta (paper §3.5, lossless under
    smooth SPS); otherwise the global argmax (needs the async barrier to stay
    lossless, paper §5.2). Zero-survival tokens sort to the tail and can only
    decrease Theta, so they are never admitted (no explicit filtering needed).

    Returns: int64 tensor [R] of verification lengths in 0..gamma.
    """
    if confidence.ndim != 2:
        raise ValueError(
            f"confidence must be [R, gamma], got {tuple(confidence.shape)}"
        )
    R, gamma = confidence.shape
    device = confidence.device
    if R == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    calibrated = calibrate_confidence(confidence, sts_temperatures)
    a = survival_probabilities(calibrated)  # [R, gamma]
    N = R * gamma

    flat = a.reshape(-1)  # [N]
    req = (
        torch.arange(R, device=device).view(R, 1).expand(R, gamma).reshape(-1)
    )  # [N] request id per flat position

    sorted_vals, sort_idx = torch.sort(flat, descending=True)  # [N]
    sorted_req = req[sort_idx]  # [N]

    sps = sps_table.to(device=device, dtype=torch.float32)
    last = sps.numel() - 1

    # Theta(m) for m = 0..N admitted tokens. index m == number admitted.
    m = torch.arange(N + 1, device=device)  # [N+1]
    tau = R + torch.cat([flat.new_zeros(1), sorted_vals.cumsum(0)])  # [N+1]
    B = (R + m).clamp(max=last)  # [N+1] batch token count, clamped into table
    theta = tau * sps[B]  # [N+1]

    if early_stop:
        # First m with Theta(m+1) <= Theta(m): m* = that m (the local peak).
        # NOTE: build the "N" fallback as a device tensor via arithmetic on an
        # existing device tensor (theta), NOT torch.tensor(N, device=device).
        # The latter materializes a host scalar -> device under the active
        # DeviceContext __torch_function__ guard, which on ROCm hangs the
        # worker (observed: all 8 ranks frozen at this line, GPU 100%).
        nonincrease = theta[1:] <= theta[:-1]  # [N]
        has_drop = nonincrease.any()
        first_drop = nonincrease.int().argmax()  # 0 if none (guarded below)
        n_fallback = torch.full_like(first_drop, N)  # device tensor, no host sync
        m_star = torch.where(has_drop, first_drop, n_fallback)
    else:
        m_star = theta.argmax()  # first global max

    # ell[r] = count of admitted (top-m_star) candidates belonging to request r.
    # Sync-free: mask by position < m_star (m_star stays a 0-dim tensor).
    admitted = torch.arange(N, device=device) < m_star  # [N] bool
    ell = torch.zeros(R, dtype=torch.long, device=device)
    ell.scatter_add_(0, sorted_req, admitted.long())
    return ell


def schedule_prefix_lengths(
    confidence: torch.Tensor,
    sps_table: torch.Tensor,
    *,
    sts_temperatures: torch.Tensor | None = None,
    early_stop: bool = _GREEDY_EARLY_STOP,
) -> list[int]:
    """Hardware-Aware Prefix Scheduler (paper Algorithm 1).

    Greedily admits draft tokens across ALL requests in descending order of
    survival probability, extending throughput Theta = tau * SPS(B) one token
    at a time, until throughput stops increasing.

    Args:
        confidence: [R, gamma] per-position confidence (raw; STS applied here if
            sts_temperatures given).
        sps_table: 1-D engine throughput profile (steps/sec) indexed by batch
            token count B. Calibrated once at engine init.
        sts_temperatures: optional [gamma] STS temperatures.
        early_stop: stop at the first throughput drop (lossless under smooth SPS;
            paper §3.5). Set False for an unconstrained global search across the
            SPS sawtooth (paper §5.2) — only lossless with the async barrier.
    Returns:
        length-R list of verification lengths ell_r in 0..gamma.
    """
    ell = schedule_prefix_lengths_tensor(
        confidence,
        sps_table,
        sts_temperatures=sts_temperatures,
        early_stop=early_stop,
    )
    return ell.tolist()  # syncs; for tests / non-hot-path callers only


def resolve_q_buckets(spec: str, max_q: int) -> list[int]:
    """Parse the DSpark CUDA-graph query-length buckets (plan Y, §17.1).

    Args:
        spec: comma-separated decode_query_len values (e.g. "1,3,6"). Empty ->
            single full bucket [max_q] (Phase-1 capture behavior).
        max_q: full verify length = mtp_k + 1 (the largest valid bucket).

    Returns:
        Sorted ascending unique buckets, each clamped to 1..max_q, always
        including max_q (the safe fallback bucket for un-quantizable steps).
    """
    out = set()
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if 1 <= v <= max_q:
            out.add(v)
    out.add(max_q)  # always keep the full bucket as fallback
    return sorted(out)


def quantize_to_bucket(q: int, buckets: list[int]) -> int:
    """Round a desired query length UP to the nearest available bucket.

    Rounding UP (never down) guarantees the chosen graph verifies at least the
    requested number of tokens -> a request is never under-verified. If q exceeds
    all buckets, returns the largest (== max_q).
    """
    for b in buckets:
        if b >= q:
            return b
    return buckets[-1]


class VerifyScheduler:
    """Hardware-Aware Prefix Scheduler for confidence-scheduled block drafting.

    Owns the per-request verify-length (ell) machinery shared by any
    confidence-scheduled block drafter (DSpark today; e.g. a future Qwen block
    drafter next). Given the draft confidence head output it picks each request's
    verify length ell_r (paper Algorithm 1), then carries that ell across
    steps keyed by req_id (continuous batching reorders the batch between steps).

    Kept sync-free on the decode hot path: record_ell fires an ASYNC D2H of
    ell to a pinned buffer and the {req_id: ell} map is materialized lazily.

    Args:
        sps_table: Pre-built SPS throughput profile (from build_sps_table).
            If None, a synthetic monotone-decreasing stub is used until
            calibration is wired in.
        sts_temperatures: Optional [gamma] STS temperatures for calibration.
        async_copy_stream: Optional CUDA stream for async D2H copies. If None,
            the async D2H path is disabled (sync fallback).
    """

    def __init__(
        self,
        sps_table: torch.Tensor | None = None,
        sts_temperatures: torch.Tensor | None = None,
        async_copy_stream: torch.cuda.Stream | None = None,
    ):
        self.sps_table = sps_table
        self.sts_temperatures = sts_temperatures
        self.async_copy_stream = async_copy_stream
        self._last_ell: torch.Tensor | None = None
        # req_id -> ell map from the PREVIOUS step's propose(), re-mapped onto the
        # next step's (possibly reordered) batch by req_id. Resolved lazily from
        # the async D2H fired by record_ell (event complete by next read).
        self._ell_map_cache: dict[int, int] | None = {}
        self._ell_pending: tuple | None = None  # (event, cpu_buf, req_ids)

    def compute_ell(self, confidence: torch.Tensor) -> torch.Tensor:
        """Run the Hardware-Aware Prefix Scheduler (paper Algorithm 1) and return
        the per-request verify length ell as an int tensor [bs].

        This ONLY computes ell — it does not touch the draft tokens. The actual
        variable-length verification (Level B) consumes ell downstream to size
        each request's verification batch, which is where the throughput win
        comes from. Kept sync-free (no .item()/.tolist()) for the decode hot path.

        Args:
            confidence: [bs, L] per-position acceptance probs.
        """
        bs, L = confidence.shape
        sps_table = self.sps_table
        if sps_table is None:
            # Synthetic monotone-decreasing SPS stub until real calibration lands.
            sps_table = torch.linspace(
                1.0, 0.1, steps=bs * (L + 1) + 1, device=confidence.device
            )
        return schedule_prefix_lengths_tensor(
            confidence.detach(),
            sps_table,
            sts_temperatures=self.sts_temperatures,
        )

    def set_last_ell(self, ell: torch.Tensor | None) -> None:
        """Stash the ell computed by this step's propose() (or None)."""
        self._last_ell = ell

    def record_ell(self, req_ids: Sequence[int]) -> None:
        """Fire an ASYNC copy of this step's ell, keyed later by req_id.

        ell was computed in propose() ordered by THIS step's decode batch. We
        save {req_id: ell} so the NEXT step can re-map it onto its own (possibly
        reordered) batch by req_id — batch position is not stable across steps
        under continuous batching.
        """
        ell = self._last_ell
        if ell is None:
            self._ell_pending = None
            self._ell_map_cache = {}
            return
        ell = ell.detach()
        # Reuse the async D2H stream if available
        copy_stream = self.async_copy_stream
        if copy_stream is None:
            # No async stream — fall back to sync copy (slower but correct).
            cpu_buf = ell.cpu()
            self._ell_map_cache = {
                req_ids[i]: int(cpu_buf[i].item()) for i in range(len(req_ids))
            }
            self._ell_pending = None
            return
        default_stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(default_stream)
            cpu_buf = ell.to("cpu", non_blocking=True)
            event.record(copy_stream)
        # Keep req_ids as a plain list snapshot (CPU-only, order-safe).
        self._ell_pending = (event, cpu_buf, list(req_ids))
        # Invalidate last step's resolved map; recomputed lazily on first read.
        self._ell_map_cache = None

    @property
    def ell_by_req(self) -> dict[int, int]:
        """Lazily materialize {req_id: ell} from the async D2H fired by
        record_ell. Syncs the (already-complete) event on first read, then
        caches so repeated reads within a step are free."""
        cache = self._ell_map_cache
        if cache is not None:
            return cache
        pending = self._ell_pending
        if pending is None:
            self._ell_map_cache = {}
            return self._ell_map_cache
        event, cpu_buf, req_ids = pending
        event.synchronize()  # long done by next step; no hot-path stall
        ell_np = cpu_buf.numpy()
        n = min(len(req_ids), ell_np.shape[0])
        self._ell_map_cache = {req_ids[i]: int(ell_np[i]) for i in range(n)}
        return self._ell_map_cache

    @ell_by_req.setter
    def ell_by_req(self, value: dict[int, int]) -> None:
        # Direct assignment (e.g. reset to {}) bypasses the pending copy.
        self._ell_map_cache = value
        self._ell_pending = None

    def ell_nonblocking(self) -> dict[int, int]:
        """Non-blocking read of the ell map for the SAME-step postprocess path
        (carried back to the scheduler as fwd_output.dspark_ell). Must NOT sync:
        record_ell just fired this step's async D2H, and forcing it here would
        re-serialize CPU on GPU — the exact stall we removed. If the copy is
        already resolved (cache present) return it; if it's still pending (this
        step's fresh copy) query without waiting; else fall back to {}.

        Correctness: the scheduler only uses this to set seq.dspark_next_ell for
        NEXT-step sizing, and the worker's own _dspark_apply_q_bucket reads the
        (fully resolved) property next step regardless — so a same-step empty
        here never under-verifies."""
        cache = self._ell_map_cache
        if cache:
            return dict(cache)
        pending = self._ell_pending
        if pending is None:
            return {}
        event, cpu_buf, req_ids = pending
        if not event.query():  # not done yet — do NOT stall the hot path
            return {}
        ell_np = cpu_buf.numpy()
        n = min(len(req_ids), ell_np.shape[0])
        resolved = {req_ids[i]: int(ell_np[i]) for i in range(n)}
        self._ell_map_cache = resolved
        return dict(resolved)
