# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10
_FP64_ONE_MINUS_EPS = (
    tl.constexpr(0.9999999999999999) if HAS_TRITON else 0.9999999999999999
)


class TemperatureSchedule(NamedTuple):
    """Persistent device buffers for the answer-phase temperature override.

    Every member is a buffer owned by `SamplingStates` or `ThinkingBudgetState`
    and allocated once at `max_num_reqs`. The kernels derive the reasoning
    phase from the committed markers, so the resolved temperature is never
    computed on the host. Per-step entropy temperature control (ReSET) is
    applied separately and does not use these buffers.
    """

    answer_temperature: torch.Tensor
    """[max_num_reqs] f32 — temperature to use in the answer phase."""
    answer_enabled: torch.Tensor
    """[max_num_reqs] i32 — non-zero if the phase override applies."""
    cached_last_start: torch.Tensor
    """[max_num_reqs] i32 — last committed reasoning start marker."""
    cached_last_end: torch.Tensor
    """[max_num_reqs] i32 — last committed natural end marker."""


_NUM_SCHEDULE_ARGS = len(TemperatureSchedule._fields)


def schedule_args(
    schedule: "TemperatureSchedule | None", fallback: torch.Tensor
) -> tuple:
    """Positional kernel arguments for `schedule`, or valid filler pointers.

    Triton needs a real pointer for every argument, including ones on branches
    its `HAS_SCHEDULE` constexpr eliminates, so the static path repeats
    `fallback` instead of passing `None`.

    Args:
        schedule: Schedule buffers, or `None` for static temperature.
        fallback: Any live tensor to use as an unread placeholder pointer.

    Returns:
        A tuple of `len(TemperatureSchedule._fields)` tensors.
    """
    if schedule is None:
        return (fallback,) * _NUM_SCHEDULE_ARGS
    return tuple(schedule)


@triton.jit
def resolve_temperature(
    req_state_idx,
    temperature_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    HAS_SCHEDULE: tl.constexpr,
):
    """Resolve the temperature for one logits row, entirely on device.

    With `HAS_SCHEDULE` false this is exactly `temperature_ptr[req]`, which
    keeps the unscheduled path byte-for-byte identical to the static kernel.
    """
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if HAS_SCHEDULE:  # noqa: SIM102 - constexpr gate must stay outside the load
        if tl.load(answer_enabled_ptr + req_state_idx) != 0:
            # The answer phase starts only once a natural end marker has fully
            # committed with no later start marker. A request that never
            # entered reasoning has last_end < 0 and keeps its static value.
            last_start = tl.load(cached_last_start_ptr + req_state_idx).to(tl.int32)
            last_end = tl.load(cached_last_end_ptr + req_state_idx).to(tl.int32)
            if last_end >= 0 and last_start <= last_end:
                temperature = tl.load(answer_temperature_ptr + req_state_idx).to(
                    tl.float32
                )
    return temperature


# Offset salt keeping the draft's Gumbel noise disjoint from the target's.
# Verification is a probability-ratio test, not a Gumbel coupling, so a proposal
# and the residual it is resampled from must not share a noise vector.
# Positions are int64 and never approach 2**30, so the streams cannot collide.
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30) if HAS_TRITON else (1 << 30)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    vocab_size,
    HAS_SCHEDULE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = resolve_temperature(
        req_state_idx,
        temperature_ptr,
        answer_temperature_ptr,
        answer_enabled_ptr,
        cached_last_start_ptr,
        cached_last_end_ptr,
        HAS_SCHEDULE=HAS_SCHEDULE,
    )
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            u = tl.minimum(u, _FP64_ONE_MINUS_EPS)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # log1p keeps the winning tail at u -> 0, where fp32 resolves it.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        finite = logits > float("-inf")
        logits = tl.where(mask & finite, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_active_rows_ptr,
    vocab_size,
    answer_temperature_ptr=None,
    answer_enabled_ptr=None,
    cached_last_start_ptr=None,
    cached_last_end_ptr=None,
    IS_DRAFTING: tl.constexpr = False,
    APPLY_TEMPERATURE: tl.constexpr = False,
    HAS_ACTIVE_ROW_LIMIT: tl.constexpr = False,
    USE_FP64: tl.constexpr = False,
    PER_TOKEN_COL: tl.constexpr = False,
    HAS_SCHEDULE: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if HAS_SCHEDULE:
        # `resolve_temperature` loads unmasked, so clamp padded rows (which
        # carry a negative req_state_idx) to a valid offset and discard the
        # result below rather than indexing out of bounds.
        safe_req_idx = tl.where(is_valid_req, req_state_idx, 0)
        resolved = resolve_temperature(
            safe_req_idx,
            temp_ptr,
            answer_temperature_ptr,
            answer_enabled_ptr,
            cached_last_start_ptr,
            cached_last_end_ptr,
            HAS_SCHEDULE=True,
        )
        temp = tl.where(is_valid_req, resolved, 0.0)

    if logits_cache_ptr is not None:
        # Store the logits *before* temperature. Dividing first would produce a
        # value that is generally not representable in the cache's dtype, forcing
        # it to be fp32. Consumers (the rejection sampler) divide by the same
        # temperature on load, which reproduces the value used below bitwise.
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        store_mask = mask
        if HAS_ACTIVE_ROW_LIMIT:
            # FULL CUDA graph replay can execute with padded request rows.
            # Those rows may carry stale req_state indices; never let them
            # overwrite cached draft logits for real requests.
            active_rows = tl.load(logits_cache_active_rows_ptr)
            store_mask = store_mask & (token_idx < active_rows)
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
            + block,
            logits,
            mask=store_mask & is_valid_req,
        )

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    return gumbel_noised_argmax(
        logits,
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_active_rows_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    HAS_ACTIVE_ROW_LIMIT: tl.constexpr,
    HAS_SCHEDULE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        logits_cache_active_rows_ptr,
        vocab_size,
        answer_temperature_ptr,
        answer_enabled_ptr,
        cached_last_start_ptr,
        cached_last_end_ptr,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
        HAS_ACTIVE_ROW_LIMIT=HAS_ACTIVE_ROW_LIMIT,
        HAS_SCHEDULE=HAS_SCHEDULE,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    logits_cache_active_rows: torch.Tensor | None = None,
    use_fp64: bool = False,
    schedule: TemperatureSchedule | None = None,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits_cache_active_rows,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        *schedule_args(schedule, temperature),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        HAS_ACTIVE_ROW_LIMIT=logits_cache_active_rows is not None,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
        HAS_SCHEDULE=schedule is not None,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
