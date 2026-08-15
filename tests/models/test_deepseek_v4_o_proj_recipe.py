# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.models.deepseek_v4.nvidia.ops import o_proj
from vllm.platforms.interface import DeviceCapability


@pytest.mark.parametrize(
    ("major", "minor", "expected_recipe", "expected_tma"),
    [
        (9, 0, (1, 128, 128), False),
        (10, 0, (1, 1, 128), True),
        (12, 0, (1, 128, 128), False),
        (12, 1, (1, 128, 128), False),
    ],
)
def test_compute_fp8_einsum_recipe(
    monkeypatch: pytest.MonkeyPatch,
    major: int,
    minor: int,
    expected_recipe: tuple[int, int, int],
    expected_tma: bool,
):
    cap = DeviceCapability(major=major, minor=minor)

    monkeypatch.setattr(
        o_proj.current_platform,
        "get_device_capability",
        lambda *args, **kwargs: cap,
        raising=False,
    )
    monkeypatch.setattr(
        o_proj.current_platform,
        "is_device_capability_family",
        lambda family, *args, **kwargs: cap.to_int() // 10 == family // 10,
        raising=False,
    )

    recipe, tma_aligned_scales = o_proj.compute_fp8_einsum_recipe()

    assert recipe == expected_recipe
    assert tma_aligned_scales is expected_tma
