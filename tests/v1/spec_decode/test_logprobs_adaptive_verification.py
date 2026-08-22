# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprobs are allowed alongside DSpark adaptive verification.

Adaptive verification compacts logits on device, so the scheduled CPU layout in
`cu_num_logits_np` misassigns logprob rows. The rejection sampler now attaches
the device `cu_num_logits`, which rides along on the existing async D2H copy, so
the request no longer has to be rejected up front.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams


def _spec_config(*, adaptive: bool) -> SpeculativeConfig:
    stub = SimpleNamespace(enable_adaptive_verification=adaptive, method="dspark")
    return cast(SpeculativeConfig, stub)


def test_logprobs_allowed_with_adaptive_verification():
    params = SamplingParams(**cast(dict[str, Any], {"logprobs": 5}))
    params._validate_spec_decode(_spec_config(adaptive=True))


def test_logprobs_allowed_without_adaptive_verification():
    params = SamplingParams(**cast(dict[str, Any], {"logprobs": 5}))
    params._validate_spec_decode(_spec_config(adaptive=False))


def _dspark_config(num_speculative_tokens: int) -> SpeculativeConfig:
    stub = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        enable_adaptive_verification=False,
        method="dspark",
    )
    return cast(SpeculativeConfig, stub)


def test_reset_rejected_with_spec_decoding_on_v1_runner():
    """ReSET + spec decode must fail validation on the V1 runner, not kill
    EngineCore in the worker.

    The V1 rejection sampler cannot resolve per-draft-position temperatures;
    request-boundary validation returns a 400 instead of a worker-side raise
    mid-step that takes the engine down.
    """
    params = SamplingParams(**cast(dict[str, Any], {"temperature_low": 0.1}))
    with pytest.raises(VLLMValidationError, match="temperature_low"):
        params._validate_spec_decode(_dspark_config(5))


def test_reset_allowed_with_spec_decoding_on_v2_runner():
    """The V2 runner resolves ReSET per draft position in its rejection
    sampler, so the request is admitted."""
    params = SamplingParams(**cast(dict[str, Any], {"temperature_low": 0.1}))
    params._validate_spec_decode(_dspark_config(5), spec_decode_supports_reset=True)


def test_reset_allowed_with_single_spec_token():
    params = SamplingParams(**cast(dict[str, Any], {"temperature_low": 0.1}))
    params._validate_spec_decode(_dspark_config(1))


def test_reset_allowed_without_spec_decoding():
    params = SamplingParams(**cast(dict[str, Any], {"temperature_low": 0.1}))
    params._validate_spec_decode(None)
