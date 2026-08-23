# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("raw_value", [-1.0, 10.5, float("inf"), True])
def test_chat_request_rejects_invalid_marker_penalty(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_marker_penalty"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_marker_penalty": raw_value,
            }
        )


def test_chat_request_forwards_marker_penalty_and_monitor():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_marker_penalty": 5.0,
            "reasoning_monitor": True,
        }
    )
    params = request.to_sampling_params(max_tokens=8, default_sampling_params={})
    assert params.reasoning_marker_penalty == 5.0
    assert params.reasoning_monitor is True


def test_chat_request_defaults_leave_controls_disabled():
    request = ChatCompletionRequest.model_validate(
        {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
    )
    params = request.to_sampling_params(max_tokens=8, default_sampling_params={})
    assert params.reasoning_marker_penalty is None
    assert params.reasoning_monitor is False


@pytest.mark.parametrize("raw_value", [-0.5, 11.0])
def test_completion_request_rejects_invalid_marker_penalty(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_marker_penalty"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "reasoning_marker_penalty": raw_value,
            }
        )


def test_completion_request_forwards_marker_penalty_and_monitor():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "reasoning_marker_penalty": 3.5,
            "reasoning_monitor": True,
        }
    )
    params = request.to_sampling_params(max_tokens=8, default_sampling_params={})
    assert params.reasoning_marker_penalty == 3.5
    assert params.reasoning_monitor is True
