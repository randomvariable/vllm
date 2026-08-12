# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("raw_value", [-2, 0.6, 10.5])
def test_chat_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking_token_budget": raw_value,
            }
        )


def test_chat_completion_request_accepts_valid_thinking_token_budget():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": 10,
        }
    )
    assert request.thinking_token_budget == 10


def test_chat_completion_request_accepts_minus_one_as_unlimited():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None


@pytest.mark.parametrize("raw_value", [True, False, float("nan"), float("inf"), -1.0])
def test_chat_completion_request_rejects_invalid_reasoning_marker_penalty(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_marker_penalty"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_marker_penalty": raw_value,
            }
        )


def test_chat_completion_request_accepts_reasoning_marker_penalty():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_marker_penalty": 2.5,
        }
    )
    assert request.reasoning_marker_penalty == 2.5


@pytest.mark.parametrize("raw_value", [0.6, 3.14, -2])
def test_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "thinking_token_budget": raw_value,
            }
        )


def test_completion_request_accepts_valid_thinking_token_budget():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": 5,
        }
    )
    assert request.thinking_token_budget == 5


def test_completion_request_accepts_minus_one_as_unlimited():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None


@pytest.mark.parametrize("raw_value", [True, False, float("nan"), float("inf"), -1.0])
def test_completion_request_rejects_invalid_reasoning_marker_penalty(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_marker_penalty"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "reasoning_marker_penalty": raw_value,
            }
        )


def test_completion_request_accepts_reasoning_marker_penalty():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "reasoning_marker_penalty": 2.5,
        }
    )
    assert request.reasoning_marker_penalty == 2.5


@pytest.mark.parametrize("raw_value", [-2, 0.6, 10.5])
def test_chat_completion_request_rejects_invalid_reasoning_answer_reserve(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_answer_reserve"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_answer_reserve": raw_value,
            }
        )


def test_chat_completion_request_accepts_reasoning_answer_reserve():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_answer_reserve": 128,
        }
    )
    assert request.reasoning_answer_reserve == 128


def test_chat_completion_request_reasoning_answer_reserve_minus_one_is_unset():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_answer_reserve": -1,
        }
    )
    assert request.reasoning_answer_reserve is None


@pytest.mark.parametrize("raw_value", [-2, 0.6, 10.5])
def test_completion_request_rejects_invalid_reasoning_answer_reserve(raw_value):
    with pytest.raises(VLLMValidationError, match="reasoning_answer_reserve"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "reasoning_answer_reserve": raw_value,
            }
        )


def test_completion_request_accepts_reasoning_answer_reserve():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "reasoning_answer_reserve": 128,
        }
    )
    assert request.reasoning_answer_reserve == 128
