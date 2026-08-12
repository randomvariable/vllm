# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import numpy as np

from vllm.v1.spec_decode.timing import SpecForwardTimings


@dataclass
class AcceptanceEstimator:
    """Exponentially weighted acceptance statistics for one load bucket.

    The decay is applied per *draft event*, never per scheduler step. A
    per-step decay would make the estimator adapt faster exactly when
    throughput is high, which is precisely when the controller reading it
    should be moving least. Decaying per draft event makes the effective
    window a fixed amount of evidence rather than a fixed amount of
    wall-clock time.

    Both the accepted and the drafted count are kept per position. This is
    not redundant: position `j` is only observable while `j < K`, so a single
    per-position rate would read zero for never-drafted positions and ratchet
    `K` down monotonically. Keeping the denominator lets `rate_at_pos` report
    `None` for "no evidence" instead of a false zero. `SpecDecodingStats`
    keeps both counts for the same reason.

    Args:
        half_life_drafts: Number of draft events after which the weight of an
            existing observation has halved. Must be positive.
        max_spec_tokens: Length of the per-position accumulators, i.e. the
            largest draft length that can ever be observed.
    """

    half_life_drafts: float
    max_spec_tokens: int
    w_drafts: float = 0.0
    w_draft_tokens: float = 0.0
    w_accepted_tokens: float = 0.0
    w_accepted_per_pos: np.ndarray = field(init=False, repr=False)
    w_drafted_per_pos: np.ndarray = field(init=False, repr=False)
    decay: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.half_life_drafts <= 0:
            raise ValueError("half_life_drafts must be positive.")
        if self.max_spec_tokens < 0:
            raise ValueError("max_spec_tokens must be non-negative.")
        self.decay = 0.5 ** (1.0 / self.half_life_drafts)
        self.w_accepted_per_pos = np.zeros(self.max_spec_tokens, dtype=np.float64)
        self.w_drafted_per_pos = np.zeros(self.max_spec_tokens, dtype=np.float64)

    def observe(self, num_draft_tokens: int, num_accepted: int) -> None:
        """Fold one draft event into the estimate.

        Args:
            num_draft_tokens: Draft tokens proposed for this request in this
                step. Zero is a no-op: nothing was drafted, so nothing was
                observed.
            num_accepted: Draft tokens accepted by the verifier, excluding the
                bonus token. Must not exceed `num_draft_tokens`.

        Raises:
            ValueError: If `num_accepted` is negative or exceeds
                `num_draft_tokens`, or `num_draft_tokens` exceeds
                `max_spec_tokens`.
        """
        if num_draft_tokens <= 0:
            return
        if num_draft_tokens > self.max_spec_tokens:
            raise ValueError(
                f"num_draft_tokens {num_draft_tokens} exceeds max_spec_tokens "
                f"{self.max_spec_tokens}."
            )
        if not 0 <= num_accepted <= num_draft_tokens:
            raise ValueError(
                f"num_accepted {num_accepted} outside [0, {num_draft_tokens}]."
            )

        decay = self.decay
        self.w_drafts = self.w_drafts * decay + 1.0
        self.w_draft_tokens = self.w_draft_tokens * decay + num_draft_tokens
        self.w_accepted_tokens = self.w_accepted_tokens * decay + num_accepted
        self.w_drafted_per_pos *= decay
        self.w_accepted_per_pos *= decay
        self.w_drafted_per_pos[:num_draft_tokens] += 1.0
        self.w_accepted_per_pos[:num_accepted] += 1.0

    @property
    def acceptance_rate(self) -> float:
        """Weighted fraction of drafted tokens that were accepted.

        Returns:
            A value in `[0, 1]`, or `0.0` when nothing has been drafted yet.
            Callers that must distinguish "no evidence" from "nothing accepted"
            should gate on `effective_n`.
        """
        if self.w_draft_tokens <= 0.0:
            return 0.0
        return self.w_accepted_tokens / self.w_draft_tokens

    @property
    def mean_accept_len(self) -> float:
        """Mean accepted tokens per draft event, including the bonus token.

        Returns:
            A value of at least `1.0`; `1.0` on a fresh estimator.
        """
        if self.w_drafts <= 0.0:
            return 1.0
        return 1.0 + self.w_accepted_tokens / self.w_drafts

    def rate_at_pos(self, j: int) -> float | None:
        """Acceptance at draft position `j`, conditional on being drafted.

        Args:
            j: Zero-based draft position.

        Returns:
            The weighted acceptance rate at `j`, or `None` when position `j`
            has never been drafted within the current window. `None` means
            "unobservable", not "zero" — see the class docstring.
        """
        if not 0 <= j < self.max_spec_tokens:
            return None
        drafted = float(self.w_drafted_per_pos[j])
        if drafted <= 0.0:
            return None
        return float(self.w_accepted_per_pos[j]) / drafted

    @property
    def effective_n(self) -> float:
        """Weighted count of draft events currently in the window.

        Returns:
            `0.0` on a fresh estimator, rising strictly with each observation
            towards `1 / (1 - decay)`. Use it as a warm-up gate: a controller
            reading a near-zero `effective_n` is reading noise.
        """
        return self.w_drafts

    def per_pos_rates(self) -> tuple[float | None, ...]:
        """All per-position rates.

        Returns:
            One entry per position in `[0, max_spec_tokens)`, each either a
            rate or `None` for never-drafted.
        """
        return tuple(self.rate_at_pos(j) for j in range(self.max_spec_tokens))


@dataclass(frozen=True)
class ForwardCostSample:
    """One batch-level forward cost measurement.

    `num_tokens` is the *forward's* token count, not the request count. That
    is what a trivial adapter must feed to `token_points` in
    `build_sps_table` (`confidence_scheduler.py:91-138`), which indexes on
    forward size; one calibration source, two consumers.

    Args:
        num_tokens: Tokens in the target forward.
        num_reqs: Requests in the batch, used as the load bucket key.
        target_ms: Target model forward duration in milliseconds.
        draft_ms: Drafter forward duration, or `None` when no drafter forward
            ran in this step.
    """

    num_tokens: int
    num_reqs: int
    target_ms: float
    draft_ms: float | None = None

    @classmethod
    def from_timings(cls, timings: SpecForwardTimings) -> "ForwardCostSample":
        """Build a sample from a worker-reported timing record.

        Args:
            timings: Timings transported from the worker.

        Returns:
            The equivalent cost sample.
        """
        return cls(
            num_tokens=timings.num_tokens,
            num_reqs=timings.num_reqs,
            target_ms=timings.target_ms,
            draft_ms=timings.draft_ms,
        )

    @property
    def total_ms(self) -> float:
        """Combined draft and target forward duration in milliseconds."""
        return self.target_ms + (self.draft_ms or 0.0)

    @property
    def steps_per_second(self) -> float:
        """Throughput of this forward, in steps per second.

        Returns:
            `1000 / total_ms`, or `0.0` for a non-positive duration. This is
            the `sps_points` domain of `build_sps_table`.
        """
        total = self.total_ms
        if total <= 0.0:
            return 0.0
        return 1000.0 / total


@dataclass(frozen=True)
class SpecDecodeSignals:
    """Frozen read interface over the telemetry.

    Primitives only, deliberately: this is the entire surface a future depth
    governor sees. Freezing it now is what keeps later work from growing six
    parallel mechanisms.

    Args:
        batch_size: Load bucket these signals were read from.
        acceptance_rate: Weighted accepted/drafted ratio in `[0, 1]`.
        mean_accept_len: Mean accepted tokens per draft, including the bonus.
        effective_n: Weighted evidence count; near zero means not warm.
        max_spec_tokens: Length of `acceptance_per_pos`.
        acceptance_per_pos: Per-position rates, `None` where never drafted.
        target_ms: Smoothed target forward duration, or `None` if unmeasured.
        draft_ms: Smoothed drafter forward duration, or `None` if unmeasured.
        forward_num_tokens: Smoothed forward token count, or `None`.
        steps_per_second: Smoothed forward throughput, or `None`.
    """

    batch_size: int
    acceptance_rate: float
    mean_accept_len: float
    effective_n: float
    max_spec_tokens: int
    acceptance_per_pos: tuple[float | None, ...]
    target_ms: float | None
    draft_ms: float | None
    forward_num_tokens: float | None
    steps_per_second: float | None


@dataclass
class _CostEstimator:
    """Exponentially weighted forward cost for one load bucket."""

    decay: float
    w: float = 0.0
    w_target_ms: float = 0.0
    w_draft_ms: float = 0.0
    w_draft_events: float = 0.0
    w_num_tokens: float = 0.0

    def observe(self, sample: ForwardCostSample) -> None:
        decay = self.decay
        self.w = self.w * decay + 1.0
        self.w_target_ms = self.w_target_ms * decay + sample.target_ms
        self.w_num_tokens = self.w_num_tokens * decay + sample.num_tokens
        self.w_draft_ms *= decay
        self.w_draft_events *= decay
        if sample.draft_ms is not None:
            self.w_draft_ms += sample.draft_ms
            self.w_draft_events += 1.0

    @property
    def target_ms(self) -> float | None:
        if self.w <= 0.0:
            return None
        return self.w_target_ms / self.w

    @property
    def draft_ms(self) -> float | None:
        if self.w_draft_events <= 0.0:
            return None
        return self.w_draft_ms / self.w_draft_events

    @property
    def num_tokens(self) -> float | None:
        if self.w <= 0.0:
            return None
        return self.w_num_tokens / self.w

    @property
    def steps_per_second(self) -> float | None:
        target_ms = self.target_ms
        if target_ms is None:
            return None
        total = target_ms + (self.draft_ms or 0.0)
        if total <= 0.0:
            return None
        return 1000.0 / total


class SpecDecodeTelemetry:
    """Fleet-level speculative decoding instrument, owned by the Scheduler.

    Answers three questions: what acceptance is being achieved, what the
    draft and target forwards cost, and at what load. It holds no policy and
    changes no behaviour.

    Per-batch-size bucketing is load-bearing, not an optimisation. `K = 0` is
    an absorbing state for a naive estimator: the acceptance update in the
    scheduler is guarded by `if scheduled_spec_token_ids`, so at `K = 0`
    nothing is observed and a governor that selected `K = 0` could never learn
    to leave it. The configured schedule in the dynamic speculative decoding
    docs ends `[129, 512, 0]`, so `K = 0` is a normal operating point.
    Bucketing means `K = 0` at `B = 200` does not erase what was learned at
    `B = 8`.

    **Control-loop dead time.** `K` is selected in `Scheduler.schedule()`
    (`scheduler.py:1192`), but the acceptance for that `K` is not observed
    until `update_from_output` (`scheduler.py:1774`) of that step. Under
    `AsyncScheduler` there is a further step of lag
    (`async_scheduler.py:23-25`). Any controller tuned as if this loop were
    delay-free will oscillate, and the oscillation will look like a policy bug
    when it is not one.

    **Constraint on any future `select_k`.** `CudagraphUtils` builds
    `decode_query_lens` from exactly the `K` values present in
    `num_speculative_tokens_per_batch_size`
    (`vllm/v1/worker/gpu/cudagraph_utils.py:197-221`). A governor emitting a
    `K` outside that set falls out of CUDA-graph replay into eager — a
    throughput cliff that gets misdiagnosed as "the policy is bad". Any
    `select_k` must be range-restricted to that set.

    Args:
        max_spec_tokens: Largest draft length that can be observed.
        half_life_drafts: Draft events after which evidence weight halves.
        max_batch_size: Largest batch size that gets its own bucket. Larger
            batch sizes share the top bucket.

    Raises:
        ValueError: If `max_batch_size` is not positive.
    """

    def __init__(
        self,
        max_spec_tokens: int,
        half_life_drafts: float = 512,
        max_batch_size: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        self.max_spec_tokens = max_spec_tokens
        self.half_life_drafts = half_life_drafts
        self.max_batch_size = max_batch_size
        # Fixed-size lists indexed directly by batch size rather than a dict:
        # exact bucket isolation, O(1) indexing with no hashing and no rehash
        # growth on the scheduler hot path. Entries are built lazily so a
        # steady-state batch size allocates once, never per observation.
        self._acceptance: list[AcceptanceEstimator | None] = [None] * (
            max_batch_size + 1
        )
        self._cost: list[_CostEstimator | None] = [None] * (max_batch_size + 1)
        self._cost_decay = 0.5 ** (1.0 / half_life_drafts)

    def _bucket(self, batch_size: int) -> int:
        if batch_size < 0:
            return 0
        return min(batch_size, self.max_batch_size)

    def observe_acceptance(
        self, batch_size: int, num_draft_tokens: int, num_accepted: int
    ) -> None:
        """Record one request's draft outcome at a given load.

        Args:
            batch_size: Requests in the batch the draft was issued under.
                Hoist this out of any per-request loop.
            num_draft_tokens: Draft tokens proposed for this request.
            num_accepted: Draft tokens accepted, excluding the bonus token.
        """
        if num_draft_tokens <= 0:
            return
        idx = self._bucket(batch_size)
        estimator = self._acceptance[idx]
        if estimator is None:
            estimator = AcceptanceEstimator(
                half_life_drafts=self.half_life_drafts,
                max_spec_tokens=self.max_spec_tokens,
            )
            self._acceptance[idx] = estimator
        estimator.observe(num_draft_tokens, num_accepted)

    def observe_forward(self, sample: ForwardCostSample) -> None:
        """Record one batch-level forward cost measurement.

        Args:
            sample: Cost sample keyed on its own `num_reqs`.
        """
        idx = self._bucket(sample.num_reqs)
        estimator = self._cost[idx]
        if estimator is None:
            estimator = _CostEstimator(decay=self._cost_decay)
            self._cost[idx] = estimator
        estimator.observe(sample)

    def snapshot(self, batch_size: int) -> SpecDecodeSignals:
        """Read the current signals for one load bucket.

        Args:
            batch_size: Load bucket to read.

        Returns:
            A frozen snapshot. A cold bucket yields `effective_n == 0.0` and
            `None` cost fields rather than fabricated values.
        """
        idx = self._bucket(batch_size)
        acceptance = self._acceptance[idx]
        cost = self._cost[idx]
        if acceptance is None:
            rate = 0.0
            mean_len = 1.0
            effective_n = 0.0
            per_pos: tuple[float | None, ...] = (None,) * self.max_spec_tokens
        else:
            rate = acceptance.acceptance_rate
            mean_len = acceptance.mean_accept_len
            effective_n = acceptance.effective_n
            per_pos = acceptance.per_pos_rates()
        return SpecDecodeSignals(
            batch_size=batch_size,
            acceptance_rate=rate,
            mean_accept_len=mean_len,
            effective_n=effective_n,
            max_spec_tokens=self.max_spec_tokens,
            acceptance_per_pos=per_pos,
            target_ms=None if cost is None else cost.target_ms,
            draft_ms=None if cost is None else cost.draft_ms,
            forward_num_tokens=None if cost is None else cost.num_tokens,
            steps_per_second=None if cost is None else cost.steps_per_second,
        )

    def cost_points(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Measured cost curve, in `build_sps_table` argument order.

        Returns:
            `(token_points, sps_points)`, one entry per warm load bucket,
            ready to pass to `build_sps_table` unmodified.
        """
        token_points: list[int] = []
        sps_points: list[float] = []
        for estimator in self._cost:
            if estimator is None:
                continue
            num_tokens = estimator.num_tokens
            sps = estimator.steps_per_second
            if num_tokens is None or sps is None:
                continue
            token_points.append(int(round(num_tokens)))
            sps_points.append(sps)
        return tuple(token_points), tuple(sps_points)
