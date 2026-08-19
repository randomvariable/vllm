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

from vllm.config.speculative import SpeculativeConfig
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
