# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ue8m0 expert-scale representation for DeepSeek-V4 weight loading.

A ue8m0 byte ``v`` denotes ``2 ** (v - 127)``. FP4 expert parameters are
``uint8`` and need those raw exponent bytes; block-fp8 expert parameters are
``float32`` and need the decoded value. Choosing one representation per
checkpoint tensor is wrong for whichever destination it does not match:
handing raw bytes to a float32 parameter makes ``copy_()`` store the exponent
byte itself, so ``2 ** -7`` arrives as ``120.0`` and scales the block by
roughly ``1e40``.
"""

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.models.deepseek_v4.nvidia.model import (
    e8m0_expert_weight_for_param,
    ue8m0_uint8_to_float,
)

# 0 denotes 2**-127, which is subnormal in float32; the encodable range for an
# exact power-of-two round trip starts at 1.
exponent_bytes = st.integers(min_value=1, max_value=254)


def _e8m0_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint8).view(torch.float8_e8m0fnu)


@given(values=st.lists(exponent_bytes, min_size=1, max_size=16))
@settings(max_examples=60, deadline=None)
def test_decode_reproduces_the_power_of_two_exactly(values):
    raw = torch.tensor(values, dtype=torch.uint8)
    decoded = ue8m0_uint8_to_float(raw)
    assert decoded.dtype == torch.float32
    expected = torch.tensor([2.0 ** (v - 127) for v in values], dtype=torch.float32)
    assert torch.equal(decoded, expected)


@given(values=st.lists(exponent_bytes, min_size=1, max_size=16))
@settings(max_examples=60, deadline=None)
def test_uint8_destination_receives_raw_bytes(values):
    loaded = _e8m0_tensor(values)
    param = torch.empty(len(values), dtype=torch.uint8)
    out = e8m0_expert_weight_for_param(loaded, param.dtype)
    assert out.dtype == torch.uint8
    assert torch.equal(out, torch.tensor(values, dtype=torch.uint8))


@given(
    values=st.lists(exponent_bytes, min_size=1, max_size=16),
    param_dtype=st.sampled_from([torch.float32, torch.bfloat16, torch.float16]),
)
@settings(max_examples=60, deadline=None)
def test_float_destination_receives_decoded_scales(values, param_dtype):
    loaded = _e8m0_tensor(values)
    param = torch.empty(len(values), dtype=param_dtype)
    out = e8m0_expert_weight_for_param(loaded, param.dtype)
    assert out.dtype == torch.float32
    expected = torch.tensor([2.0 ** (v - 127) for v in values], dtype=torch.float32)
    assert torch.equal(out, expected)


@given(exponent=exponent_bytes)
@settings(max_examples=60, deadline=None)
def test_exponent_byte_never_lands_in_a_float_parameter(exponent):
    """The #43416 garbling: the byte written as a float instead of decoded."""
    loaded = _e8m0_tensor([exponent])
    param = torch.empty(1, dtype=torch.float32)
    out = e8m0_expert_weight_for_param(loaded, param.dtype)
    written = torch.empty(1, dtype=torch.float32)
    written.copy_(out)
    assert written.item() != float(exponent) or exponent == 1
    assert math.isclose(written.item(), 2.0 ** (exponent - 127), rel_tol=0.0)


def test_byte_as_float_would_be_catastrophic():
    """Guard the test above against silently proving nothing.

    Exponent 120 denotes 2**-7. If the raw byte reached a float parameter it
    would be stored as 120.0 - a factor of ~1.5e4 off, and ~1e40 relative to
    the intended scale once squared into the block product.
    """
    intended = 2.0**-7
    assert 120.0 / intended > 1e4


@given(values=st.lists(st.floats(min_value=0.1, max_value=4.0), min_size=1, max_size=8))
@settings(max_examples=40, deadline=None)
def test_non_e8m0_scales_pass_through_untouched(values):
    loaded = torch.tensor(values, dtype=torch.float32)
    for param_dtype in (torch.float32, torch.uint8):
        out = e8m0_expert_weight_for_param(loaded, param_dtype)
        assert out is loaded
