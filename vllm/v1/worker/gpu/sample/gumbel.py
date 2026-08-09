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


class TemperatureSchedule(NamedTuple):
    """Persistent device buffers for step-aware temperature resolution.

    Every member is a buffer owned by `SamplingStates`, `RequestState` or
    `ThinkingBudgetState` and allocated once at `max_num_reqs`. The kernels
    derive the per-row step count, the anneal progress and the reasoning phase
    from them, so a resolved temperature is never computed on the host.
    """

    temperature_final: torch.Tensor
    """[max_num_reqs] f32 — temperature the anneal converges to."""
    anneal_steps: torch.Tensor
    """[max_num_reqs] i32 — output tokens the anneal spans."""
    schedule_enabled: torch.Tensor
    """[max_num_reqs] i32 — non-zero if the request anneals."""
    answer_temperature: torch.Tensor
    """[max_num_reqs] f32 — temperature to use in the answer phase."""
    answer_enabled: torch.Tensor
    """[max_num_reqs] i32 — non-zero if the phase override applies."""
    expanded_local_pos: torch.Tensor
    """[num_tokens] i32 — row offset within its request's logits."""
    total_len: torch.Tensor
    """[max_num_reqs] i32 — prompt_len + committed output length."""
    prompt_len: torch.Tensor
    """[max_num_reqs] i32 — prompt length, subtracted to get the step."""
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
    token_idx,
    req_state_idx,
    temperature_ptr,
    temperature_final_ptr,
    anneal_steps_ptr,
    schedule_enabled_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    expanded_local_pos_ptr,
    total_len_ptr,
    prompt_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    HAS_SCHEDULE: tl.constexpr,
):
    """Resolve the temperature for one logits row, entirely on device.

    With `HAS_SCHEDULE` false this is exactly `temperature_ptr[req]`, which
    keeps the unscheduled path byte-for-byte identical to the static kernel.
    """
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if HAS_SCHEDULE:
        if tl.load(schedule_enabled_ptr + req_state_idx) != 0:
            # generated_steps is derived on device; the host stages no counter.
            total_len = tl.load(total_len_ptr + req_state_idx).to(tl.int32)
            prompt_len = tl.load(prompt_len_ptr + req_state_idx).to(tl.int32)
            local_pos = tl.load(expanded_local_pos_ptr + token_idx).to(tl.int32)
            step = total_len - prompt_len + local_pos
            if step < 0:
                step = 0
            anneal_steps = tl.load(anneal_steps_ptr + req_state_idx).to(tl.int32)
            if anneal_steps > 0:
                progress = step.to(tl.float32) / anneal_steps.to(tl.float32)
                progress = tl.minimum(progress, 1.0)
                final = tl.load(temperature_final_ptr + req_state_idx).to(tl.float32)
                temperature = temperature + progress * (final - temperature)

        if tl.load(answer_enabled_ptr + req_state_idx) != 0:
            # The answer phase starts only once a natural end marker has fully
            # committed with no later start marker. A request that never
            # entered reasoning has last_end < 0 and keeps its scheduled value.
            last_start = tl.load(cached_last_start_ptr + req_state_idx).to(tl.int32)
            last_end = tl.load(cached_last_end_ptr + req_state_idx).to(tl.int32)
            if last_end >= 0 and last_start <= last_end:
                temperature = tl.load(answer_temperature_ptr + req_state_idx).to(
                    tl.float32
                )
    return temperature


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    temperature_final_ptr,
    anneal_steps_ptr,
    schedule_enabled_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    expanded_local_pos_ptr,
    total_len_ptr,
    prompt_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_SCHEDULE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = resolve_temperature(
        token_idx,
        req_state_idx,
        temperature_ptr,
        temperature_final_ptr,
        anneal_steps_ptr,
        schedule_enabled_ptr,
        answer_temperature_ptr,
        answer_enabled_ptr,
        expanded_local_pos_ptr,
        total_len_ptr,
        prompt_len_ptr,
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
    schedule: TemperatureSchedule | None = None,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        *schedule_args(schedule, temperature),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SCHEDULE=schedule is not None,
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
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    vocab_size,
    temperature_final_ptr=None,
    anneal_steps_ptr=None,
    schedule_enabled_ptr=None,
    answer_temperature_ptr=None,
    answer_enabled_ptr=None,
    expanded_local_pos_ptr=None,
    total_len_ptr=None,
    prompt_len_ptr=None,
    cached_last_start_ptr=None,
    cached_last_end_ptr=None,
    APPLY_TEMPERATURE: tl.constexpr = False,
    USE_FP64: tl.constexpr = False,
    PER_TOKEN_COL: tl.constexpr = False,
    HAS_SCHEDULE: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    # Clamp so the schedule loads below stay in bounds for padded rows; the
    # resolved value is discarded for them anyway.
    safe_req_idx = tl.where(is_valid_req, req_state_idx, 0)
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if HAS_SCHEDULE:
        resolved = resolve_temperature(
            token_idx,
            safe_req_idx,
            temp_ptr,
            temperature_final_ptr,
            anneal_steps_ptr,
            schedule_enabled_ptr,
            answer_temperature_ptr,
            answer_enabled_ptr,
            expanded_local_pos_ptr,
            total_len_ptr,
            prompt_len_ptr,
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
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32–1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        if USE_FP64:
            u = tl_rand64(gumbel_seed, block, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, block, includes_zero=False)
            # Draw the large-noise tail (which decides the argmax winner) from u -> 0,
            # where fp32 has fine resolution, instead of u -> 1, where fp32 spacing is
            # ~2**-24. The naive `-log(-log(u))` puts the winning tail at u -> 1,
            # hard-capping the noise at ~16.6 and coarsely quantizing it; using
            # `log1p(-u)` == `log(1 - u)` keeps the tail in the well-resolved region.
            # Note `1 - u` would lose precision for small u, so `log1p` is required.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    temperature_final_ptr,
    anneal_steps_ptr,
    schedule_enabled_ptr,
    answer_temperature_ptr,
    answer_enabled_ptr,
    expanded_local_pos_ptr,
    total_len_ptr,
    prompt_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
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
        logits_cache_stride,
        logits_cache_col_ptr,
        vocab_size,
        temperature_final_ptr,
        anneal_steps_ptr,
        schedule_enabled_ptr,
        answer_temperature_ptr,
        answer_enabled_ptr,
        expanded_local_pos_ptr,
        total_len_ptr,
        prompt_len_ptr,
        cached_last_start_ptr,
        cached_last_end_ptr,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
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
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
    schedule: TemperatureSchedule | None = None,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
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
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        *schedule_args(schedule, temperature),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
        HAS_SCHEDULE=schedule is not None,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
