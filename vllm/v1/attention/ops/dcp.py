# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DCP collective selection and direct symmetric-memory implementations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.cp_common import (
    DirectCPWorkspace,
    direct_cp_enabled,
    direct_cp_multicast_enabled,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator


# LSE/output combine


def mask_dcp_empty_shards_(
    lse: torch.Tensor,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
) -> None:
    if seq_lens is None and query_start_loc is None:
        return
    if seq_lens is None or query_start_loc is None:
        raise ValueError("seq_lens and query_start_loc must be provided together")
    if (
        seq_lens.ndim != 1
        or query_start_loc.ndim != 1
        or query_start_loc.shape[0] != seq_lens.shape[0] + 1
    ):
        raise ValueError("query_start_loc must contain one boundary per sequence")

    row_indices = torch.arange(
        lse.shape[0], device=lse.device, dtype=query_start_loc.dtype
    )
    sequence_indices = torch.searchsorted(
        query_start_loc[1:], row_indices, right=True
    ).clamp_max(seq_lens.shape[0] - 1)
    empty_rows = (row_indices >= query_start_loc[-1]) | (
        seq_lens[sequence_indices] == 0
    )
    lse.masked_fill_(empty_rows[:, None], float("-inf"))


# AG + RS/AR implementation


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    valid_n_offsets = num_n_offsets < N

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    lse = tl.load(
        lses_ptr + lse_offsets,
        mask=valid_n_offsets,
        other=-float("inf"),
    ).to(tl.float32)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset).to(tl.float32)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {
        "HEAD_DIM": D,
        "N": N,
        "N_ROUNDED": next_power_of_2(N),
        "IS_BASE_E": is_lse_base_on_e,
    }
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def _cp_lse_common(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    cp_attn_lse = cp_attn_lse.contiguous()
    mask_dcp_empty_shards_(cp_attn_lse, seq_lens, query_start_loc)
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    head_major_output: bool = False,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    if head_major_output:
        out = cp_group.reduce_scatter_head_major(out, dim=1)
    else:
        out = cp_group.reduce_scatter(out, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_rs_into(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    output_provider: Callable[[torch.Tensor], torch.Tensor],
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
):
    """Correct DCP partials and reduce-scatter into borrowed output storage."""
    if cp_group.world_size <= 1:
        raise RuntimeError("cp_lse_ag_out_rs_into requires DCP world size > 1")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cp_lse_ag_out_rs_into is eager-only")

    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    output = output_provider(out)
    if not isinstance(output, torch.Tensor):
        raise TypeError("DCP output provider must return a tensor")
    out = cp_group.reduce_scatter_into(out, output, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    out = cp_group.all_reduce(out)

    if return_lse:
        return out, lse
    return out




def get_dcp_workspace_max_num_tokens(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    speculative_tokens = vllm_config.num_speculative_tokens
    tokens_per_seq = (
        1
        + (
            2
            if speculative_config is not None and speculative_config.parallel_drafting
            else 1
        )
        * speculative_tokens
    )
    return min(
        scheduler_config.max_num_batched_tokens,
        max(
            scheduler_config.max_num_seqs * tokens_per_seq,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )


def reserve_query_head_storage(
    query: torch.Tensor, padded_num_heads: int
) -> torch.Tensor:
    """Reserve backing storage for fixed-head decode kernels."""
    assert query.ndim == 3
    assert query.shape[1] <= padded_num_heads
    padded = query.new_empty((query.shape[0], padded_num_heads, query.shape[2]))
    padded.resize_(query.shape)
    padded.copy_(query)
    return padded


# Symmetric-memory A2A implementation


_A2A_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


class DirectDCPA2AWorkspace(DirectCPWorkspace):
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _A2A_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                f"Direct DCP A2A requires at least one ubatch slot, got {num_ubatches}"
            )
        super().__init__(group, device, num_ubatches)
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            output_shape, dtype
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(lse_shape, torch.float32)
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        output = partial_output.new_empty(
            (num_tokens, self.heads_per_rank, self.head_dim)
        )
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.peer_output_ptrs[ubatch],
            self.peer_lse_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_output[ubatch],
            self.received_lse[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output,
            self.world_size,
            self.rank,
            self.max_num_tokens,
            is_lse_base_on_e,
        )
        return output


@functools.cache
def get_direct_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace | None:
    if not direct_cp_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_A2A, _A2A_SUPPORTED_DTYPES
    ):
        return None
    return DirectDCPA2AWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )


# Q gather

# Symmetric-memory implementation


def _q_gather_layout_supported(
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    padded_num_heads: int | None,
) -> bool:
    element_size = torch.empty((), dtype=dtype).element_size()
    gathered_num_heads = world_size * heads_per_rank
    storage_num_heads = (
        gathered_num_heads if padded_num_heads is None else padded_num_heads
    )
    return (
        heads_per_rank * head_dim * element_size % 16 == 0
        and storage_num_heads * head_dim * element_size % 16 == 0
    )


class DirectDCPQGatherWorkspace(DirectCPWorkspace):
    """Publish query shards directly into the consumer-final symmetric buffer.

    The final buffer is reusable after the downstream DCP output combine. That
    combine orders all ranks after attention has consumed the gathered query.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
        padded_num_heads: int | None = None,
    ) -> None:
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP q-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_num_tokens < 1 or heads_per_rank < 1 or head_dim < 1:
            raise ValueError(
                "Direct DCP q-gather dimensions must be positive, got "
                f"T={max_num_tokens}, H={heads_per_rank}, D={head_dim}"
            )
        gathered_num_heads = group.size() * heads_per_rank
        if not _q_gather_layout_supported(
            group.size(), heads_per_rank, head_dim, dtype, padded_num_heads
        ):
            raise ValueError("Direct DCP q-gather requires 16-byte-aligned query rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP q-gather requires at least two ranks")
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.gathered_num_heads = gathered_num_heads
        self.padded_num_heads = (
            self.gathered_num_heads if padded_num_heads is None else padded_num_heads
        )
        if self.padded_num_heads < self.gathered_num_heads:
            raise ValueError(
                "Direct DCP q-gather padded heads must cover gathered heads: "
                f"{self.padded_num_heads} < {self.gathered_num_heads}"
            )
        self.head_dim = head_dim

        query_shape = (
            num_ubatches,
            max_num_tokens,
            self.padded_num_heads,
            head_dim,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.final_query, _ = self._allocate(query_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        query_multicast_ptrs = self._multicast_ptrs(self.final_query)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(query_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(
            query_ptr and signal_ptr for query_ptr, signal_ptr in self.multicast_ptrs
        ):
            raise RuntimeError(
                "Direct DCP q-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 1))
        torch.accelerator.synchronize()

    def gather(self, local_query: torch.Tensor) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP q-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        if local_query.ndim == 3 and local_query.shape[1] != self.heads_per_rank:
            raise ValueError(
                f"DCP q-gather expected {self.heads_per_rank} local query heads, "
                f"got {local_query.shape[1]}"
            )

        num_tokens = local_query.shape[0]
        output = torch.as_strided(
            self.final_query[ubatch],
            size=(num_tokens, self.gathered_num_heads, self.head_dim),
            stride=(
                self.gathered_num_heads * self.head_dim,
                self.head_dim,
                1,
            ),
        )
        query_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_q_gather(
            local_query,
            output,
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            self.world_size,
            self.rank,
            self.max_num_tokens,
            self.padded_num_heads,
            query_multicast_ptr,
            signal_multicast_ptr,
        )
        return output


@functools.cache
def get_direct_dcp_q_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
    padded_num_heads: int | None = None,
) -> DirectDCPQGatherWorkspace | None:
    if not direct_cp_multicast_enabled(group, dtype, envs.VLLM_USE_DIRECT_DCP_Q_GATHER):
        return None
    if not _q_gather_layout_supported(
        group.world_size, heads_per_rank, head_dim, dtype, padded_num_heads
    ):
        return None
    return DirectDCPQGatherWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
        padded_num_heads,
    )


# KV gather

# Symmetric-memory implementation


_KV_GATHER_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _kv_gather_layout_supported(token_dim: int, dtype: torch.dtype) -> bool:
    return token_dim * torch.empty((), dtype=dtype).element_size() % 16 == 0


class DirectDCPKVGatherWorkspace(DirectCPWorkspace):
    """Persistent symmetric buffers for direct DCP KV gather."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_gathered_tokens: int,
        token_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _KV_GATHER_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP kv-gather does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP kv-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_gathered_tokens < 1 or token_dim < 1:
            raise ValueError(
                "Direct DCP kv-gather dimensions must be positive, got "
                f"T={max_gathered_tokens}, D={token_dim}"
            )
        if not _kv_gather_layout_supported(token_dim, dtype):
            raise ValueError("Direct DCP kv-gather requires 16-byte-aligned KV rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP kv-gather requires at least two ranks")
        if max_gathered_tokens % self.world_size != 0:
            raise ValueError(
                "Direct DCP kv-gather capacity must divide evenly across "
                f"ranks: {max_gathered_tokens} % {self.world_size} != 0"
            )
        self.max_gathered_tokens = max_gathered_tokens

        kv_shape = (num_ubatches, 2, max_gathered_tokens, token_dim)
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_kv, _ = self._allocate(kv_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        kv_multicast_ptrs = self._multicast_ptrs(self.received_kv)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(kv_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(kv_ptr and signal_ptr for kv_ptr, signal_ptr in self.multicast_ptrs):
            raise RuntimeError(
                "Direct DCP kv-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 2))
        torch.accelerator.synchronize()

    def gather(self, gathered_kv: torch.Tensor, local_kv: torch.Tensor) -> None:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        kv_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            gathered_kv,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
            kv_multicast_ptr,
            signal_multicast_ptr,
        )


@functools.cache
def get_direct_dcp_kv_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    token_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPKVGatherWorkspace | None:
    if not direct_cp_multicast_enabled(
        group,
        dtype,
        envs.VLLM_USE_DIRECT_DCP_KV_GATHER,
        _KV_GATHER_SUPPORTED_DTYPES,
    ):
        return None
    if not _kv_gather_layout_supported(token_dim, dtype):
        return None
    return DirectDCPKVGatherWorkspace(
        group.device_group,
        device,
        max_gathered_tokens,
        token_dim,
        dtype,
        num_ubatches,
    )


# MLA DCP backend selection


class DCPCombine(Protocol):
    def __call__(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor: ...


class MLADCPManager:
    """Select and own layer-level collective implementations for MLA DCP."""

    _kv_gather: Callable[[torch.Tensor, torch.Tensor], object]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        num_heads: int,
        query_head_dim: int,
        output_head_dim: int,
        query_dtype: torch.dtype,
        output_dtype: torch.dtype,
        padded_num_heads: int | None,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.group = get_dcp_group()
        self.device = torch.device(device)
        self.num_ubatches = max(parallel_config.num_ubatches, 1)
        self.max_num_tokens = get_dcp_workspace_max_num_tokens(vllm_config)
        self.use_a2a = parallel_config.dcp_comm_backend == "a2a"
        self.padded_num_heads = padded_num_heads

        self.combine = self._init_combine(
            num_heads,
            output_head_dim,
            output_dtype,
            is_lse_base_on_e,
            use_pcp,
        )
        self.query_gather = (
            None
            if use_pcp
            else self._init_query_gather(
                num_heads,
                query_head_dim,
                query_dtype,
            )
        )

    def _init_combine(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> DCPCombine:
        direct_workspace = None
        if self.use_a2a:
            direct_workspace = get_direct_dcp_a2a_workspace(
                self.group,
                self.device,
                self.max_num_tokens,
                num_heads,
                head_dim,
                dtype,
                self.num_ubatches,
            )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP A2A for MLA.")
            return functools.partial(
                self._direct_workspace_combine,
                direct_workspace,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        # Deferred: dcp_alltoall imports mask_dcp_empty_shards_ from this module.
        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
        return functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _direct_workspace_combine(
        self,
        direct_workspace: DirectDCPA2AWorkspace,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Forced MQA path pass all batch tokens (including prefill) into combine,
        # which may exceed the direct symmetric-memory workspace. Fall back to
        # the nccl a2a combine for those cases.
        if partial_output.shape[0] <= direct_workspace.max_num_tokens:
            return direct_workspace.lse_reduce(
                partial_output,
                partial_lse,
                is_lse_base_on_e,
                seq_lens,
                query_start_loc,
            )
        return dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )

    def _init_query_gather(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        direct_workspace = get_direct_dcp_q_gather_workspace(
            self.group,
            self.device,
            self.max_num_tokens,
            num_heads,
            head_dim,
            dtype,
            self.num_ubatches,
            self.padded_num_heads,
        )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP query gather for MLA.")
            return functools.partial(
                self._direct_workspace_query_gather,
                direct_workspace,
            )
        return self._gather_query

    def _gather_query(self, query: torch.Tensor) -> torch.Tensor:
        query = self.group.all_gather(query, dim=1)
        if self.padded_num_heads is not None:
            query = reserve_query_head_storage(query, self.padded_num_heads)
        return query

    def _direct_workspace_query_gather(
        self,
        direct_workspace: DirectDCPQGatherWorkspace,
        query: torch.Tensor,
    ) -> torch.Tensor:
        # Forced MQA path can be taken for long sparse MLA prefills, whose batch size
        # may exceed the direct symmetric-memory workspace. Fall back to allgather
        # for those cases.
        if query.shape[0] <= direct_workspace.max_num_tokens:
            return direct_workspace.gather(query)
        return self._gather_query(query)

    def init_kv_gather(
        self,
        workspace: torch.Tensor,
        max_gathered_tokens: int,
    ) -> None:
        world_size = self.group.world_size
        assert max_gathered_tokens > 0
        assert max_gathered_tokens % world_size == 0
        assert workspace.ndim == 2
        assert workspace.is_contiguous()
        assert workspace.shape[0] == (
            max_gathered_tokens + max_gathered_tokens // world_size
        )
        assert workspace.shape[1] > 0

        direct_workspace = get_direct_dcp_kv_gather_workspace(
            self.group,
            workspace.device,
            max_gathered_tokens,
            workspace.shape[1],
            workspace.dtype,
            self.num_ubatches,
        )
        if direct_workspace is not None:
            logger.info_once(
                "Using direct symmetric-memory DCP chunked-context KV gather for MLA."
            )
            self._kv_gather = direct_workspace.gather
        else:
            self._kv_gather = functools.partial(
                torch.distributed.all_gather_into_tensor,
                group=self.group.device_group,
            )

    def kv_gather(
        self,
        gathered_kv: torch.Tensor,
        local_kv: torch.Tensor,
    ) -> object:
        return self._kv_gather(gathered_kv, local_kv)
