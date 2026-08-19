# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM workspaces must not move once a CUDA graph could have captured them.

A graph captures the raw device pointer of the buffer current at capture time.
``Tensor.resize_()`` frees the old storage and moves the buffer, so a
post-capture grow makes every replay dereference memory the caching allocator
has already handed to other tensors.
"""

import sys
import types

import pytest
import torch

from vllm.utils.flashinfer import presize_flashinfer_gemm_workspaces


@pytest.fixture
def stub_flashinfer(monkeypatch):
    """Install a fake ``flashinfer.utils`` recording _get_cache_buf calls."""
    calls: list[dict] = []

    def _get_cache_buf(name, size, device, zero_init=None):
        call = {"name": name, "size": size, "device": device}
        if zero_init is not None:
            call["zero_init"] = zero_init
        calls.append(call)
        return torch.empty(0)

    utils_mod = types.SimpleNamespace(_get_cache_buf=_get_cache_buf)
    root = types.SimpleNamespace(utils=utils_mod)
    monkeypatch.setitem(sys.modules, "flashinfer", root)
    monkeypatch.setitem(sys.modules, "flashinfer.utils", utils_mod)
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: True)
    return calls


def test_noop_without_flashinfer(monkeypatch):
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: False)
    monkeypatch.setitem(sys.modules, "flashinfer.utils", None)
    presize_flashinfer_gemm_workspaces(torch.device("cpu"))


def test_presize_requests_zeroed_buffer(stub_flashinfer):
    device = torch.device("cpu")
    presize_flashinfer_gemm_workspaces(device, size=1234)

    assert stub_flashinfer, "expected at least one workspace to be pre-grown"
    for call in stub_flashinfer:
        assert call["size"] == 1234
        assert call["device"] is device
        # cuDNN split-K plans keep completion semaphores in the workspace and
        # spin forever on garbage, so the buffer must be requested zeroed.
        assert call.get("zero_init") is True


def test_presize_falls_back_when_zero_init_unsupported(monkeypatch):
    """Older FlashInfer releases have no zero_init parameter."""
    calls: list[tuple] = []

    def _get_cache_buf(name, size, device):
        calls.append((name, size, device))
        return torch.empty(0)

    utils_mod = types.SimpleNamespace(_get_cache_buf=_get_cache_buf)
    monkeypatch.setitem(sys.modules, "flashinfer.utils", utils_mod)
    monkeypatch.setattr("vllm.utils.flashinfer.has_flashinfer", lambda: True)

    presize_flashinfer_gemm_workspaces(torch.device("cpu"), size=99)

    assert [c[1] for c in calls] == [99] * len(calls)
    assert calls, "fallback path must still pre-grow the workspace"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SM100Workspace allocates on cuda in its constructor",
)
def test_growing_the_mla_workspace_retires_rather_than_moves():
    from vllm.v1.attention.backends.mla.cutlass_mla import SM100Workspace

    workspace = SM100Workspace(1024)
    original = workspace.get_buf()
    original_ptr = original.data_ptr()

    workspace._grow_to(4096)

    grown = workspace.get_buf()
    assert grown.shape[0] >= 4096
    assert grown.data_ptr() != original_ptr
    # The old generation must still be alive at its original address: a graph
    # captured before the grow replays against exactly this pointer.
    assert original in workspace._retired_bufs
    assert original.data_ptr() == original_ptr
    # Split-KV workspaces carry semaphore/accumulator regions that kernels
    # expect zeroed; a recycled dirty block would hang or corrupt them.
    assert bool((grown == 0).all())
