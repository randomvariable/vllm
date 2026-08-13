# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local smoke: a ReSET request and a plain request both generate.

The ReSET request travels the batched ReSET logits processor (entropy-threshold
temperature, arXiv 2606.13233) while the plain request stays on the static
path. Both completing is the observable contract; ReSET must not stall or empty
generation. ReSET requires ``temperature=1.0`` so the per-token temperature is
applied once, by the processor.
"""

import os

# Force the V1 model runner before vLLM resolves the config: ReSET on V1
# travels the batched logits-processor path.
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import pytest  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def llm():
    return LLM(
        model=MODEL,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=512,
    )


def test_reset_and_plain_requests_complete(llm):
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


def test_reset_rejects_non_unit_temperature():
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError):
        SamplingParams(temperature=0.8, temperature_low=0.1, max_tokens=8)
