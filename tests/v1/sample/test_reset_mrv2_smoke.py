# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local smoke: ReSET on the V2 model runner (gfx1151).

The V2 runner bypasses vLLM logits processors, so ReSET is resolved inside the
V2 sampler from the same on-device core. This forces the V2 runner and checks a
ReSET request and a plain request both generate. Applied once, on device, with
no per-token host sync.
"""

import os

import pytest

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def llm():
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return LLM(
        model=MODEL,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=512,
    )


def test_reset_and_plain_requests_complete_v2(llm):
    plain = SamplingParams(temperature=0.8, max_tokens=24, seed=0)
    reset = SamplingParams(
        temperature=1.0,
        temperature_low=0.1,
        temperature_high=1.0,
        entropy_threshold=0.5505,
        reset_window=32,
        max_tokens=24,
        seed=0,
    )

    out_plain = llm.generate(["The capital of France is"], plain)
    out_reset = llm.generate(
        ["Compute 17 times 24 step by step, then give the answer."], reset
    )

    assert out_plain[0].outputs[0].token_ids
    assert out_reset[0].outputs[0].token_ids
