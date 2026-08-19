# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_BATCH_INVARIANT must not silently promise determinism it cannot give.

Speculative decoding draws the acceptance coin from the same RNG stream as
proposal sampling (`rejection_sampler_utils` uses `tl_rand32(seed, pos)`, and
`gumbel` uses `tl.randint(seed, pos)`), separated only by position offset. Until
the per-domain streams and preemption recovery from upstream #52522 land, a
seeded request can change its output when batch size, request order or
preemption changes, so the flag must say so rather than appear to hold.
"""

import pytest

from vllm.config.speculative import speculative_batch_invariance_unsupported_reason


def test_no_speculative_config_is_supported():
    assert speculative_batch_invariance_unsupported_reason(None) is None


@pytest.mark.parametrize("method", ["dspark", "dflash", "eagle3", "mtp"])
def test_speculative_methods_report_a_reason(method):
    reason = speculative_batch_invariance_unsupported_reason(method)
    assert reason, f"{method} must report why batch invariance cannot hold"
    assert "52522" in reason
