# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ROCm platform detection."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_rocm_device_count_cache():
    from vllm.platforms.rocm import _rocm_device_count_stateless

    _rocm_device_count_stateless.cache_clear()
    yield
    _rocm_device_count_stateless.cache_clear()


def test_rocm_device_count_returns_amdsmi_zero_without_hip_fallback():
    from vllm.platforms.rocm import _rocm_device_count_stateless

    with (
        patch("torch.cuda._is_compiled", return_value=True),
        patch("torch.cuda._device_count_amdsmi", return_value=0),
        patch("torch._C._cuda_getDeviceCount") as hip_count,
    ):
        assert _rocm_device_count_stateless() == 0
        hip_count.assert_not_called()


def test_rocm_device_count_falls_back_when_amdsmi_reports_negative_count():
    from vllm.platforms.rocm import _rocm_device_count_stateless

    with (
        patch("torch.cuda._is_compiled", return_value=True),
        patch("torch.cuda._device_count_amdsmi", return_value=-1),
        patch("torch._C._cuda_getDeviceCount", return_value=1) as hip_count,
    ):
        assert _rocm_device_count_stateless() == 1
        hip_count.assert_called_once_with()


def test_rocm_device_count_uses_positive_amdsmi_count():
    from vllm.platforms.rocm import _rocm_device_count_stateless

    with (
        patch("torch.cuda._is_compiled", return_value=True),
        patch("torch.cuda._device_count_amdsmi", return_value=1),
        patch("torch._C._cuda_getDeviceCount") as hip_count,
    ):
        assert _rocm_device_count_stateless() == 1
        hip_count.assert_not_called()


def test_rocm_device_count_cache_keys_are_isolated():
    from vllm.platforms.rocm import _rocm_device_count_stateless

    with (
        patch("torch.cuda._is_compiled", return_value=True),
        patch("torch.cuda._device_count_amdsmi", side_effect=[1, 2, 3]) as amdsmi_count,
        patch("torch._C._cuda_getDeviceCount") as hip_count,
    ):
        assert _rocm_device_count_stateless(None) == 1
        assert _rocm_device_count_stateless("") == 2
        assert _rocm_device_count_stateless("0") == 3
        assert _rocm_device_count_stateless(None) == 1
        assert _rocm_device_count_stateless("") == 2
        assert _rocm_device_count_stateless("0") == 3

        assert amdsmi_count.call_count == 3
        hip_count.assert_not_called()
