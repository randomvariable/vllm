# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

_MGTB_SAMPLING_CONTROL_FIELDS = (
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "max_tokens",
    "min_tokens",
    "bad_words",
    "thinking_token_budget",
    "temperature_low",
    "temperature_high",
    "entropy_threshold",
    "reset_window",
    "reasoning_answer_temperature",
    "reasoning_marker_penalty",
    "repetition_detection",
    "structured_outputs",
    "logit_bias",
    "allowed_token_ids",
    "extra_args",
)


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if callable(value):
        return {
            "callable": f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        }
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "value": str(value),
    }


def _sampling_control_stack(sampling_params: object | None) -> object:
    if sampling_params is None:
        return None
    return {
        name: _canonical_value(getattr(sampling_params, name, None))
        for name in _MGTB_SAMPLING_CONTROL_FIELDS
    }


def build_mgtb_calibration_key(
    model_config: object,
    quant_config: object | None,
    reasoning_config: object | None,
    sampling_params: object | None = None,
) -> str:
    """Build a stable key for model, tokenizer, quantization, and controls."""

    def identity(config: object | None, *names: str) -> str | None:
        if config is None:
            return None
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return str(value)
        return None

    marker_token_ids: tuple[int, ...] = ()
    if reasoning_config is not None:
        try:
            marker_token_ids = tuple(
                getattr(reasoning_config, "marker_token_ids", ()) or ()
            )
        except (TypeError, ValueError):
            marker_token_ids = ()
    control_stack: dict[str, object] = {
        "marker_token_ids": marker_token_ids,
        "mgtb_schema": 1,
        "sampling_policy": _sampling_control_stack(sampling_params),
    }
    payload = {
        "schema_version": 1,
        "model": identity(model_config, "model"),
        "model_revision": identity(model_config, "revision"),
        "tokenizer": identity(model_config, "tokenizer"),
        "tokenizer_revision": identity(model_config, "tokenizer_revision"),
        "quantization": identity(quant_config, "name", "quantization")
        or (type(quant_config).__name__ if quant_config is not None else None),
        "control_stack": control_stack,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _load_calibration_file(path: str) -> MGTBCalibration | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None

        key = payload.get("key")
        raw_scores = payload.get("scores_by_bucket")
        raw_controls = payload.get("control_stack", {})
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(raw_scores, dict)
            or not isinstance(raw_controls, dict)
            or not raw_scores
        ):
            return None

        scores: dict[int, tuple[float, ...]] = {}
        for bucket, values in raw_scores.items():
            if not isinstance(values, list) or not values:
                return None
            bucket_scores = tuple(float(score) for score in values)
            if any(not math.isfinite(score) for score in bucket_scores):
                return None
            scores[int(bucket)] = bucket_scores

        return MGTBCalibration(
            key=key,
            scores_by_bucket=scores,
            control_stack=raw_controls,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ):
        return None


@dataclass(frozen=True)
class MGTBCalibration:
    """Immutable upper-tail score samples keyed by position bucket."""

    key: str
    scores_by_bucket: Mapping[int, tuple[float, ...]]
    control_stack: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {
            int(bucket): tuple(float(score) for score in scores)
            for bucket, scores in self.scores_by_bucket.items()
        }
        if (
            not self.key
            or not normalized
            or any(not values for values in normalized.values())
        ):
            raise ValueError("MGT-B calibration artifact has no score samples")
        if any(
            not math.isfinite(score)
            for values in normalized.values()
            for score in values
        ):
            raise ValueError("MGT-B calibration scores must be finite")
        object.__setattr__(self, "scores_by_bucket", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "control_stack",
            MappingProxyType(dict(self.control_stack)),
        )

    @classmethod
    def load(
        cls, path: str | Path, expected_key: str | None = None
    ) -> MGTBCalibration | None:
        """Load a cached schema-checked artifact, failing closed on mismatch."""
        calibration = _load_calibration_file(str(Path(path)))
        if calibration is None:
            return None
        if expected_key is not None and calibration.key != expected_key:
            return None
        return calibration

    def upper_tail_probability(
        self, score: float, position: int, bucket_size: int = 512
    ) -> float:
        """Return a smoothed empirical upper-tail probability for ``score``."""
        bucket = position // bucket_size
        samples = self.scores_by_bucket.get(bucket)
        if not samples:
            samples = self.scores_by_bucket.get(-1, ())
        if not samples:
            return 1.0
        exceedances = sum(sample >= score for sample in samples)
        return (1.0 + exceedances) / (1.0 + len(samples))


@dataclass(frozen=True)
class MGTBConfig:
    """Host-side MGT-B signal and recurrence configuration."""

    window: int = 64
    stride: int = 32
    ngram_sizes: tuple[int, ...] = (6, 7, 8)
    weights: tuple[float, ...] = (0.15, 0.10, 0.20, 0.35, 0.18, 0.02)
    beta_exponents: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    position_bucket_size: int = 512
    threshold: float = 10.0
    calibration: MGTBCalibration | None = None

    def __post_init__(self) -> None:
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("MGT-B window and stride must be positive")
        if not self.ngram_sizes or any(n <= 0 for n in self.ngram_sizes):
            raise ValueError("MGT-B requires positive n-gram sizes")
        if len(self.weights) != 6:
            raise ValueError("MGT-B requires six signal weights")
        if not self.beta_exponents:
            raise ValueError("MGT-B requires at least one beta exponent")
        if self.position_bucket_size <= 0:
            raise ValueError("MGT-B position bucket size must be positive")


@dataclass
class MGTBRequestState:
    """Per-request MGT-B trajectory, window features, and recurrence state."""

    prompt_token_ids: Sequence[int] = ()
    config: MGTBConfig = field(default_factory=MGTBConfig)
    sampled_tokens: int = 0
    emitted_tokens: int = 0
    deleted_tokens: int = 0
    statistic: float = 0.0
    alarms: int = 0
    refractory_until: int = 0
    tokens: list[int] = field(default_factory=list, init=False)
    entropies: list[float] = field(default_factory=list, init=False)
    logprobs: list[float] = field(default_factory=list, init=False)
    window_features: list[tuple[float, ...]] = field(default_factory=list, init=False)
    window_scores: list[float] = field(default_factory=list, init=False)
    window_probabilities: list[float] = field(default_factory=list, init=False)
    window_factors: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.prompt_token_ids = tuple(
            int(token_id) for token_id in self.prompt_token_ids
        )
        self._prompt_ngrams = {
            (n, tuple(self.prompt_token_ids[index : index + n]))
            for n in self.config.ngram_sizes
            for index in range(len(self.prompt_token_ids) - n + 1)
        }

    @property
    def last_features(self) -> tuple[float, ...] | None:
        """Return the most recent six-dimensional signal, if available."""
        return self.window_features[-1] if self.window_features else None

    def advance(
        self,
        entropy: float,
        logprob: float,
        drift: float = 0.0,
        threshold: float | None = None,
        refractory_tokens: int | None = None,
        *,
        token_id: int | None = None,
        position: int | None = None,
        emitted: bool = True,
    ) -> bool:
        """Consume one committed observation and return whether it alarms."""
        self.sampled_tokens += 1
        if emitted:
            self.emitted_tokens += 1
        self.tokens.append(-1 if token_id is None else int(token_id))
        self.entropies.append(float(entropy) if math.isfinite(float(entropy)) else 0.0)
        self.logprobs.append(float(logprob) if math.isfinite(float(logprob)) else 0.0)
        if position is None:
            position = len(self.tokens) - 1
        if len(self.tokens) < self.config.window:
            return False
        if (len(self.tokens) - self.config.window) % self.config.stride:
            return False

        features = self._compute_features()
        score = float(np.dot(np.asarray(features), np.asarray(self.config.weights)))
        probability = self._upper_tail_probability(score, position)
        factor = self._evidence_factor(probability)
        self.window_features.append(features)
        self.window_scores.append(score)
        self.window_probabilities.append(probability)
        self.window_factors.append(factor)

        threshold = self.config.threshold if threshold is None else threshold
        refractory_tokens = (
            self.config.stride if refractory_tokens is None else refractory_tokens
        )
        self.statistic = max(0.0, self.statistic) + math.log(factor) - drift
        if self.statistic < threshold or position < self.refractory_until:
            return False
        self.alarms += 1
        self.refractory_until = position + max(0, refractory_tokens)
        return True

    def record_sampled(self, count: int) -> None:
        """Account for sampled tokens without a committed observation."""
        self.sampled_tokens += max(0, int(count))

    def record_emitted(self, count: int) -> None:
        """Account for committed output tokens without monitor observables."""
        self.emitted_tokens += max(0, int(count))

    def record_deleted(self, count: int) -> None:
        """Account for sampled tokens discarded by speculation or rollback."""
        count = max(0, int(count))
        self.deleted_tokens += count
        self.sampled_tokens += count

    def _compute_features(self) -> tuple[float, ...]:
        window_start = len(self.tokens) - self.config.window
        window_end = len(self.tokens)
        entropy_window = self.entropies[window_start:window_end]
        logprob_window = self.logprobs[window_start:window_end]
        entropy_mean = float(np.mean(entropy_window))
        negative_logprob_mean = -float(np.mean(logprob_window))
        repetition, confidence_increase = self._ngram_features(window_start, window_end)

        if window_start == 0:
            local_entropy_change = 0.0
            negative_local_entropy_change = 0.0
        else:
            prefix_mean = max(float(np.mean(self.entropies[:window_start])), 1e-8)
            log_ratio = math.log(max(entropy_mean, 1e-8) / prefix_mean)
            local_entropy_change = max(0.0, log_ratio)
            negative_local_entropy_change = max(0.0, -log_ratio)
        return (
            entropy_mean,
            negative_logprob_mean,
            repetition,
            confidence_increase,
            local_entropy_change,
            negative_local_entropy_change,
        )

    def _ngram_features(
        self, window_start: int, window_end: int
    ) -> tuple[float, float]:
        repeat_count = 0
        eligible_count = 0
        max_confidence_increase = 0.0
        for n in self.config.ngram_sizes:
            if n > window_end:
                continue
            occurrences: dict[tuple[int, ...], list[tuple[int, float]]] = {}
            for start in range(window_end - n + 1):
                gram = tuple(self.tokens[start : start + n])
                if min(gram, default=-1) < 0 or (n, gram) in self._prompt_ngrams:
                    continue
                mean_logprob = float(np.mean(self.logprobs[start : start + n]))
                occurrences.setdefault(gram, []).append((start, mean_logprob))
            for gram_occurrences in occurrences.values():
                prior: list[float] = []
                for start, mean_logprob in gram_occurrences:
                    if not (window_start <= start < window_end - n + 1):
                        prior.append(mean_logprob)
                        continue
                    eligible_count += 1
                    if prior:
                        repeat_count += 1
                        max_confidence_increase = max(
                            max_confidence_increase,
                            mean_logprob - max(prior),
                        )
                    prior.append(mean_logprob)
        if eligible_count == 0:
            return 0.0, 0.0
        return repeat_count / eligible_count, max(0.0, max_confidence_increase)

    def _upper_tail_probability(self, score: float, position: int) -> float:
        calibration = self.config.calibration
        if calibration is None:
            return 1.0
        bucket = position // self.config.position_bucket_size
        samples = calibration.scores_by_bucket.get(bucket)
        if not samples:
            samples = calibration.scores_by_bucket.get(-1, ())
        if not samples:
            return 1.0
        exceedances = sum(sample >= score for sample in samples)
        return (1.0 + exceedances) / (1.0 + len(samples))

    def _evidence_factor(self, probability: float) -> float:
        probability = min(max(probability, 1e-6), 1.0)
        factors = [
            beta * probability ** (beta - 1.0) for beta in self.config.beta_exponents
        ]
        return max(float(np.mean(factors)), 1e-12)
