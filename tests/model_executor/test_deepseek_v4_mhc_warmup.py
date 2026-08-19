# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-size selection for the DeepSeek V4 mHC TileLang warmup.

The tilelang mHC kernels derive their split-k factor from two different K
dimensions: ``mhc_pre_tilelang`` and ``mhc_fused_post_pre_tilelang`` use
``hc_mult * hidden_size``, while ``mhc_pre_broadcast_tilelang`` (the b12x mHC
path) uses ``hidden_size`` alone. Warmup sweeps only the larger K, which is
sufficient because ``split_k = min(num_sms // grid, K // 64 // 4)`` is
non-increasing in ``grid``: the smaller K lowers the cap, which merges
plateaus rather than creating new ones, so the broadcast transitions are a
subset of the swept ones. These tests pin that relationship, so a future
change to either the sweep or a kernel's K dimension cannot silently leave a
kernel to JIT-compile during serving.
"""

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _compute_mhc_pre_num_split,
    _select_mhc_warmup_token_sizes,
)

HIDDEN_SIZE = 2048
HC_MULT = 4
NUM_SMS = 48
MAX_TOKENS = 16_384


def _transitions(*, hidden_size: int, hc_mult: int, num_sms: int) -> list[int]:
    """Token counts where split-k changes, computed independently of warmup."""
    sizes: list[int] = []
    last = None
    grid = 1
    while (size := (grid - 1) * 64 + 1) <= MAX_TOKENS:
        split = _compute_mhc_pre_num_split(
            num_tokens=size,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            num_sms=num_sms,
        )
        if split != last:
            sizes.append(size)
            last = split
        grid += 1
    return sizes


def test_broadcast_transitions_are_subset_of_swept_transitions():
    """Broadcast K (hc_mult=1) must not need transitions the sweep misses."""
    swept = set(_transitions(hidden_size=HIDDEN_SIZE, hc_mult=HC_MULT, num_sms=NUM_SMS))
    broadcast = set(_transitions(hidden_size=HIDDEN_SIZE, hc_mult=1, num_sms=NUM_SMS))
    assert broadcast <= swept


def test_selected_sizes_cover_both_kernel_families():
    selected = set(
        _select_mhc_warmup_token_sizes(
            max_tokens=MAX_TOKENS,
            cudagraph_capture_sizes=[],
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            num_sms=NUM_SMS,
        )
    )
    for hc_mult in (HC_MULT, 1):
        missing = [
            size
            for size in _transitions(
                hidden_size=HIDDEN_SIZE, hc_mult=hc_mult, num_sms=NUM_SMS
            )
            if size not in selected
        ]
        assert not missing, f"hc_mult={hc_mult} transitions uncovered: {missing}"


def test_small_fma_decode_sizes_are_warmed():
    """mhc_fused_post_pre_tilelang takes a distinct grid for num_tokens < 8."""
    selected = set(
        _select_mhc_warmup_token_sizes(
            max_tokens=MAX_TOKENS,
            cudagraph_capture_sizes=[],
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            num_sms=NUM_SMS,
        )
    )
    assert {1, 2, 4}.issubset(selected)
    assert 8 in selected


def test_selection_respects_max_tokens_and_includes_capture_sizes():
    selected = _select_mhc_warmup_token_sizes(
        max_tokens=512,
        cudagraph_capture_sizes=[7, 96, 1024],
        hidden_size=HIDDEN_SIZE,
        hc_mult=HC_MULT,
        num_sms=NUM_SMS,
    )
    assert selected == sorted(set(selected))
    assert max(selected) <= 512
    assert {7, 96}.issubset(selected)
    assert 1024 not in selected


def test_non_positive_max_tokens_selects_nothing():
    assert (
        _select_mhc_warmup_token_sizes(
            max_tokens=0,
            cudagraph_capture_sizes=[64],
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            num_sms=NUM_SMS,
        )
        == []
    )
