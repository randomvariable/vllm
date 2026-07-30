# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the sliding-window block skip in the Triton paged-decode kernel.

``kernel_paged_attention_2d`` starts its KV loop at the window edge::

    start_block = max(seq_len - SLIDING_WINDOW, 0) // BLOCK_SIZE

instead of block 0, so strictly out-of-window K/V is never loaded. The claim
under test is a SHAPE claim: post-change decode latency should be roughly flat
in context length, because the work becomes window-bounded rather than
context-bounded.

Both arms run ``_kernel_paged_attention_2d_benchmark`` below, which is a copy of
the production kernel with one added constexpr, ``SKIP_OUT_OF_WINDOW``, that
selects the loop start. Everything else -- crucially the token-granularity
score mask -- is identical, so both arms compute the SAME output and we are
timing two correct paths rather than one correct path and one broken one.

Two correctness gates run before timing (``--check`` / on by default):

1. the ``SKIP_OUT_OF_WINDOW=True`` arm against the real, imported production
   kernel, which proves this copy has not drifted from the kernel it claims to
   measure;
2. the ``False`` arm against the ``True`` arm, which proves the "before" bound
   is a correct path and not a faster wrong answer.

Usage:
    python benchmarks/kernels/benchmark_sliding_window_paged_decode.py
    python benchmarks/kernels/benchmark_sliding_window_paged_decode.py \
        --seq-lens 8192 --sliding-window 512 --arm after --profile-iters 20

On gfx1151 the host venv cannot launch kernels. Run it inside the runtime
image the same way ``homelab/rocm-dev-test.sh`` does -- same docker flags and
same overlay of changed ``vllm/`` sources onto the installed package -- but
invoking this script instead of pytest.

Note: ``rocprofv3 --kernel-trace`` works in that image, but counter collection
(``-i`` with a ``pmc:`` line, e.g. FETCH_SIZE) hangs indefinitely on gfx1151 and
had to be abandoned, so memory-traffic counters are not available here.
"""

import os
import statistics
import sys

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

logger = init_logger(__name__)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _kernel_paged_attention_2d_benchmark(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    SLIDING_WINDOW: tl.constexpr,  # int
    SKIP_OUT_OF_WINDOW: tl.constexpr,  # bool -- the arm under test
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
):
    """Copy of ``kernel_paged_attention_2d`` parametrised on the loop start.

    Reduced to the code paths this benchmark exercises: bf16/fp16 KV, no
    ALiBi, no sinks, no FP8 output. ``SKIP_OUT_OF_WINDOW`` selects the loop
    start; the score mask below is untouched, so both settings agree bitwise.
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if cur_batch_query_len > 1:
        return

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # The arm under test. SKIP_OUT_OF_WINDOW=True is current behaviour;
    # False reproduces the pre-change bound, reading from block 0.
    if SKIP_OUT_OF_WINDOW and SLIDING_WINDOW > 0:
        start_block = tl.maximum(seq_len - SLIDING_WINDOW, 0) // BLOCK_SIZE
    else:
        start_block = 0

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    for j in range(start_block, num_blocks):
        start_n = j * BLOCK_SIZE
        abs_token_idx = start_n + offs_n
        kv_load_mask = abs_token_idx < seq_len
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & kv_load_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & kv_load_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1

        # Token-granularity window mask, identical in both arms. This is what
        # makes the two loop bounds produce the same output.
        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    acc = acc / (L[:, None] + 1e-10)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


class DecodeCase:
    """One decode configuration and its device-side tensors."""

    def __init__(
        self,
        num_seqs: int,
        seq_len: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        sliding_window: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.num_seqs = num_seqs
        self.seq_len = seq_len
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device

        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.scale = float(1.0 / (head_size**0.5))
        # bf16/fp16 pack 16 bytes along the innermost K axis.
        self.x = 16 // torch.tensor([], dtype=dtype).element_size()

        blocks_per_seq = (seq_len + block_size - 1) // block_size
        num_blocks = num_seqs * blocks_per_seq

        self.query = torch.empty(
            num_seqs, num_query_heads, head_size, dtype=dtype, device=device
        )
        self.query.uniform_(-self.scale, self.scale)

        k = torch.empty(
            num_blocks,
            num_kv_heads,
            head_size // self.x,
            block_size,
            self.x,
            dtype=dtype,
            device=device,
        )
        k.uniform_(-self.scale, self.scale)
        v = torch.empty(
            num_blocks,
            num_kv_heads,
            head_size,
            block_size,
            dtype=dtype,
            device=device,
        )
        v.uniform_(-self.scale, self.scale)
        self.key_cache = k
        self.value_cache = v

        # Shuffled physical blocks: a real cache is not laid out in logical
        # order, and contiguous blocks would understate load cost.
        block_ids = torch.randperm(num_blocks, device=device).to(torch.int32)
        self.block_table = block_ids.view(num_seqs, blocks_per_seq)

        self.seq_lens = torch.full(
            (num_seqs,), seq_len, dtype=torch.int32, device=device
        )
        self.query_start_loc = torch.arange(
            num_seqs + 1, dtype=torch.int32, device=device
        )
        self.output = torch.empty_like(self.query)

    @property
    def start_block(self) -> int:
        if self.sliding_window <= 0:
            return 0
        return max(self.seq_len - self.sliding_window, 0) // self.block_size

    @property
    def num_blocks_per_seq(self) -> int:
        return (self.seq_len + self.block_size - 1) // self.block_size

    def _bench_kwargs(self, out: torch.Tensor, skip_out_of_window: bool) -> dict:
        num_queries_per_kv_padded = max(
            triton.next_power_of_2(self.num_queries_per_kv), 16
        )
        return dict(
            output_ptr=out,
            query_ptr=self.query,
            key_cache_ptr=self.key_cache,
            value_cache_ptr=self.value_cache,
            block_tables_ptr=self.block_table,
            seq_lens_ptr=self.seq_lens,
            scale=self.scale,
            num_query_heads=self.num_query_heads,
            num_queries_per_kv=self.num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=self.block_table.stride(0),
            query_stride_0=self.query.stride(0),
            query_stride_1=self.query.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=self.block_size,
            PHYSICAL_BLOCK_SIZE=self.block_size,
            HEAD_SIZE=self.head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(self.head_size),
            SLIDING_WINDOW=self.sliding_window,
            SKIP_OUT_OF_WINDOW=skip_out_of_window,
            x=self.x,
            stride_k_cache_0=self.key_cache.stride(0),
            stride_k_cache_1=self.key_cache.stride(1),
            stride_k_cache_2=self.key_cache.stride(2),
            stride_k_cache_3=self.key_cache.stride(3),
            stride_k_cache_4=self.key_cache.stride(4),
            stride_v_cache_0=self.value_cache.stride(0),
            stride_v_cache_1=self.value_cache.stride(1),
            stride_v_cache_2=self.value_cache.stride(2),
            stride_v_cache_3=self.value_cache.stride(3),
            query_start_len_ptr=self.query_start_loc,
        )

    def launch(self, skip_out_of_window: bool, out: torch.Tensor | None = None):
        out = self.output if out is None else out
        _kernel_paged_attention_2d_benchmark[(self.num_seqs, self.num_kv_heads)](
            **self._bench_kwargs(out, skip_out_of_window)
        )
        return out

    def launch_compiled(self, skip_out_of_window: bool):
        """Launch the benchmark copy and return the CompiledKernel handle."""
        return _kernel_paged_attention_2d_benchmark[(self.num_seqs, self.num_kv_heads)](
            **self._bench_kwargs(self.output, skip_out_of_window)
        )

    def _production_kwargs(self, out: torch.Tensor) -> dict:
        num_queries_per_kv_padded = max(
            triton.next_power_of_2(self.num_queries_per_kv), 16
        )
        one = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        return dict(
            output_ptr=out,
            query_ptr=self.query,
            key_cache_ptr=self.key_cache,
            value_cache_ptr=self.value_cache,
            sink_ptr=None,
            block_tables_ptr=self.block_table,
            seq_lens_ptr=self.seq_lens,
            alibi_slopes_ptr=None,
            scale=self.scale,
            k_scale=one,
            v_scale=one,
            out_scale_inv=1.0,
            num_query_heads=self.num_query_heads,
            num_queries_per_kv=self.num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=self.block_table.stride(0),
            query_stride_0=self.query.stride(0),
            query_stride_1=self.query.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=self.block_size,
            PHYSICAL_BLOCK_SIZE=self.block_size,
            HEAD_SIZE=self.head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(self.head_size),
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=self.sliding_window,
            x=self.x,
            stride_k_cache_0=self.key_cache.stride(0),
            stride_k_cache_1=self.key_cache.stride(1),
            stride_k_cache_2=self.key_cache.stride(2),
            stride_k_cache_3=self.key_cache.stride(3),
            stride_k_cache_4=self.key_cache.stride(4),
            stride_v_cache_0=self.value_cache.stride(0),
            stride_v_cache_1=self.value_cache.stride(1),
            stride_v_cache_2=self.value_cache.stride(2),
            stride_v_cache_3=self.value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=self.query_start_loc,
            USE_SINKS=False,
            USE_FP8=False,
        )

    def launch_kernel(self, kernel, out: torch.Tensor | None = None):
        """Launch any kernel with the production signature; return the handle.

        Used to compile a kernel taken from git history alongside the installed
        one, so their register allocations are directly comparable.
        """
        out = torch.empty_like(self.query) if out is None else out
        return kernel[(self.num_seqs, self.num_kv_heads)](
            **self._production_kwargs(out)
        )

    def launch_production(self, out: torch.Tensor | None = None):
        """Run the real production kernel, for the fidelity gate."""
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
            kernel_paged_attention_2d,
        )

        out = torch.empty_like(self.query) if out is None else out
        self.launch_kernel(kernel_paged_attention_2d, out)
        return out

    def launch_production_compiled(self):
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
            kernel_paged_attention_2d,
        )

        return self.launch_kernel(kernel_paged_attention_2d)


def check_equivalence(case: DecodeCase, atol: float) -> None:
    """Gate both arms before timing them.

    Fidelity: the benchmark copy with SKIP_OUT_OF_WINDOW=True must agree with
    the imported production kernel, or this file is measuring something else.
    Equality: the False arm must agree with the True arm, or the "before"
    number is a wrong answer computed quickly.
    """
    after = case.launch(skip_out_of_window=True, out=torch.empty_like(case.query))
    production = case.launch_production()
    torch.testing.assert_close(
        after,
        production,
        atol=atol,
        rtol=0,
        msg=lambda m: f"benchmark copy has drifted from the production kernel\n{m}",
    )

    before = case.launch(skip_out_of_window=False, out=torch.empty_like(case.query))
    torch.testing.assert_close(
        before,
        after,
        atol=atol,
        rtol=0,
        msg=lambda m: f"the two loop bounds disagree, so one arm is wrong\n{m}",
    )


def time_arm(
    case: DecodeCase, skip_out_of_window: bool, warmup: int, iters: int
) -> float:
    """Median kernel latency in microseconds."""
    fn = lambda: case.launch(skip_out_of_window)  # noqa: E731
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=iters, return_mode="median")
    return ms * 1000.0


def time_interleaved(
    first, second, warmup: int, iters: int, rounds: int
) -> tuple[float, float]:
    """Time two callables alternately, returning their per-arm medians.

    Shared by the copy's two arms and by the real pre/post-change kernels; see
    ``time_both_arms`` for why alternating matters on this hardware.
    """
    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(
            triton.testing.do_bench(
                first, warmup=warmup, rep=iters, return_mode="median"
            )
            * 1000.0
        )
        b.append(
            triton.testing.do_bench(
                second, warmup=warmup, rep=iters, return_mode="median"
            )
            * 1000.0
        )
    return statistics.median(a), statistics.median(b)


def time_both_arms(
    case: DecodeCase, warmup: int, iters: int, rounds: int
) -> tuple[float, float]:
    """Time both arms interleaved, returning ``(before_us, after_us)``.

    Timing one arm fully and then the other lets any monotonic drift during the
    run -- on this hardware, DVFS ramping the shader clock off its ~600 MHz
    floor -- land entirely on the second arm and masquerade as speedup. The
    symptom is a configuration where nothing is skipped reporting a speedup
    other than 1.00x. Alternating the arms across several rounds and taking the
    per-arm median cancels drift to first order; the ``skipped == 0`` rows then
    serve as a control that says whether it worked.
    """
    before: list[float] = []
    after: list[float] = []
    for _ in range(rounds):
        before.append(time_arm(case, False, warmup, iters))
        after.append(time_arm(case, True, warmup, iters))
    return statistics.median(before), statistics.median(after)


@torch.inference_mode()
def run_sweep(args) -> int:
    set_random_seed(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    device = args.device

    print(
        f"device={torch.cuda.get_device_name(0)}  dtype={args.dtype}  "
        f"num_seqs={args.batch_size}  q_heads={args.num_query_heads}  "
        f"kv_heads={args.num_kv_heads}  "
        f"gqa={args.num_query_heads // args.num_kv_heads}:1  "
        f"head_size={args.head_size}  block_size={args.block_size}"
    )
    print()

    header = (
        f"{'window':>7} {'ctx':>7} {'blocks':>7} {'skipped':>11} "
        f"{'before us':>11} {'after us':>10} {'speedup':>8}"
    )

    failures = 0
    for window in args.sliding_window:
        print(f"--- sliding_window = {window} ---")
        print(header)
        print("-" * len(header))
        measured: list[tuple[int, float, float]] = []
        controls: list[tuple[int, float, float, int]] = []
        for seq_len in args.seq_lens:
            case = DecodeCase(
                num_seqs=args.batch_size,
                seq_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                sliding_window=window,
                dtype=dtype,
                device=device,
            )

            if args.check:
                try:
                    check_equivalence(case, args.atol)
                except AssertionError as exc:
                    print(f"{window:>7} {seq_len:>7}   CORRECTNESS FAILURE: {exc}")
                    failures += 1
                    del case
                    torch.cuda.empty_cache()
                    continue

            before, after = time_both_arms(case, args.warmup, args.iters, args.rounds)

            nblocks = case.num_blocks_per_seq
            skipped = case.start_block
            pct = 100.0 * skipped / nblocks if nblocks else 0.0
            print(
                f"{window:>7} {seq_len:>7} {nblocks:>7} "
                f"{skipped:>5} ({pct:>3.0f}%) "
                f"{before:>11.1f} {after:>10.1f} {before / after:>7.2f}x"
            )
            measured.append((seq_len, before, after))
            if skipped == 0:
                controls.append((seq_len, before, after, skipped))
            del case
            torch.cuda.empty_cache()

        # The shape claim: 'after' should be near-flat in ctx while 'before'
        # grows with it. Growth factors make that visible without a plot.
        if len(measured) > 1:
            first, last = measured[0], measured[-1]
            print(
                f"  shape over ctx {first[0]}->{last[0]}: "
                f"before grows {last[1] / first[1]:.2f}x, "
                f"after grows {last[2] / first[2]:.2f}x"
            )

        # Control rows: where nothing is skipped both arms run the identical
        # loop, so a speedup far from 1.00x means the measurement drifted (GPU
        # clock ramp, or another process on the device) and the whole column is
        # biased. Better to say so than to report the number.
        for seq_len, before, after, skipped in controls:
            ratio = before / after
            if not 0.95 <= ratio <= 1.05:
                print(
                    f"  WARNING: ctx={seq_len} skips no blocks (skipped={skipped}) "
                    f"so both arms do identical work, but measured {ratio:.2f}x. "
                    f"This run is biased -- suspect GPU clock drift or a "
                    f"competing process; re-run on an idle device."
                )
        print()

    return failures


def _load_prechange_kernel(repo_root: str):
    """Import the pre-change kernel from git history.

    The register question is about the *production* kernel across the two
    revisions, not about the parametrised copy above: the copy shares one
    Triton source, so both of its arms could in principle allocate alike for
    reasons that do not hold for the real thing. So fetch the actual pre-change
    source out of git and compile that.

    Returns ``(kernel, revision)``, or ``(None, reason)`` if history is
    unavailable (e.g. a shallow clone or an export with no .git).
    """
    import importlib.util
    import subprocess
    import tempfile

    rel = "vllm/v1/attention/ops/chunked_prefill_paged_decode.py"
    try:
        # The commit that introduced the skip; its parent is the "before" tree.
        rev = subprocess.run(
            [
                "git",
                "-C",
                repo_root,
                "log",
                "-1",
                "--format=%H",
                "-S",
                "start_block",
                "--",
                rel,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not rev:
            return None, "could not find the commit that introduced start_block"
        src = subprocess.run(
            ["git", "-C", repo_root, "show", f"{rev}^:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return None, f"git history unavailable: {exc}"

    if "start_block" in src:
        return None, f"{rev[:9]}^ still mentions start_block; wrong parent"

    # Triton requires @jit functions to live in a real file on disk, and the
    # file must outlive this call, so it is deliberately not auto-deleted.
    fd, path = tempfile.mkstemp(suffix="_prechange.py", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(src)

    # Load it *as* a submodule of the real package: the file carries relative
    # imports (``from .prefix_prefill import ...``) that only resolve if the
    # module's package is the genuine one.
    name = "vllm.v1.attention.ops._prechange_paged_decode"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the run
        sys.modules.pop(name, None)
        return None, f"pre-change source failed to import: {exc}"
    return mod.kernel_paged_attention_2d, f"{rev[:9]}^"


@torch.inference_mode()
def run_registers(args) -> int:
    """Report VGPR allocation per kernel revision and SLIDING_WINDOW value.

    Settles whether a layer with no sliding window pays for the added
    ``if SLIDING_WINDOW > 0`` guard. ``SLIDING_WINDOW`` is a ``tl.constexpr``,
    so at 0 the branch should fold away at compile time and the post-change
    kernel should allocate exactly what the pre-change kernel did. Triton
    reports ``n_regs`` from the compiled binary, so this is the compiler's own
    number rather than an inference from timing.
    """
    set_random_seed(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    prechange, rev = _load_prechange_kernel(repo_root)
    print(f"device={torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    if prechange is None:
        print(f"pre-change kernel NOT loaded ({rev}); reporting post-change only")
    else:
        print(f"pre-change kernel loaded from {rev}")
    print()

    rows: list[tuple[str, int, int, int]] = []
    for window in args.sliding_window:
        # ctx == window means no block is skippable, isolating the branch cost
        # from the work saving.
        case = DecodeCase(
            num_seqs=args.batch_size,
            seq_len=args.seq_lens[0],
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            sliding_window=window,
            dtype=dtype,
            device=args.device,
        )
        if prechange is not None:
            k = case.launch_kernel(prechange)
            rows.append(("pre-change  (production)", window, k.n_regs, k.n_spills))
        k = case.launch_production_compiled()
        rows.append(("post-change (production)", window, k.n_regs, k.n_spills))
        k = case.launch_compiled(skip_out_of_window=False)
        rows.append(("benchmark copy, skip=off", window, k.n_regs, k.n_spills))
        k = case.launch_compiled(skip_out_of_window=True)
        rows.append(("benchmark copy, skip=on", window, k.n_regs, k.n_spills))
        del case
        torch.cuda.empty_cache()

    print(f"{'kernel':>26} {'SLIDING_WINDOW':>15} {'VGPRs':>7} {'spills':>7}")
    print("-" * 60)
    for name, window, regs, spills in rows:
        print(f"{name:>26} {window:>15} {regs:>7} {spills:>7}")
    print()

    # Registers are a compile-time proxy for a runtime cost, so also time the
    # two real kernels head to head. At SLIDING_WINDOW=0 this is the question
    # that matters: does a non-windowed layer actually run slower?
    if prechange is not None and args.time_registers:
        print(
            f"{'SLIDING_WINDOW':>15} {'pre-change us':>14} "
            f"{'post-change us':>15} {'ratio':>7}"
        )
        print("-" * 55)
        for window in args.sliding_window:
            case = DecodeCase(
                num_seqs=args.batch_size,
                seq_len=args.seq_lens[0],
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                sliding_window=window,
                dtype=dtype,
                device=args.device,
            )
            pre_us, post_us = time_interleaved(
                lambda c=case: c.launch_kernel(prechange),
                lambda c=case: c.launch_production(),
                args.warmup,
                args.iters,
                args.rounds,
            )
            print(
                f"{window:>15} {pre_us:>14.1f} {post_us:>15.1f} "
                f"{pre_us / post_us:>6.2f}x"
            )
            del case
            torch.cuda.empty_cache()
        print()

    # The verdict, computed rather than eyeballed.
    by_key = {(n, w): r for n, w, r, _ in rows}
    for window in args.sliding_window:
        pre = by_key.get(("pre-change  (production)", window))
        post = by_key.get(("post-change (production)", window))
        if pre is None or post is None:
            continue
        if pre == post:
            print(
                f"SLIDING_WINDOW={window}: unchanged at {post} VGPRs -- "
                f"the guard costs no registers here."
            )
        else:
            print(
                f"SLIDING_WINDOW={window}: {pre} -> {post} VGPRs "
                f"({post - pre:+d}) -- the guard costs registers here."
            )
    return 0


@torch.inference_mode()
def run_profile(args) -> int:
    """Single-arm, fixed-iteration run for an external profiler.

    rocprofv3 wants one hot kernel and a deterministic dispatch count, not a
    sweep with two arms and a do_bench autotune loop mixed in.

    ``--arm prechange`` / ``--arm production`` launch the real kernel from
    either side of the change instead of the parametrised copy, so a profiler
    reads register counts off the kernel that actually ships.
    """
    set_random_seed(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    seq_len = args.seq_lens[0]
    window = args.sliding_window[0]

    case = DecodeCase(
        num_seqs=args.batch_size,
        seq_len=seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        sliding_window=window,
        dtype=dtype,
        device=args.device,
    )

    if args.arm == "prechange":
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        kernel, rev = _load_prechange_kernel(repo_root)
        if kernel is None:
            print(f"cannot profile the pre-change arm: {rev}", file=sys.stderr)
            return 1
        label = f"prechange({rev})"
        launch = lambda: case.launch_kernel(kernel)  # noqa: E731
    elif args.arm == "production":
        label = "production"
        launch = case.launch_production
    else:
        skip = args.arm == "after"
        label = f"copy(skip={skip})"
        launch = lambda: case.launch(skip)  # noqa: E731

    # Compile and warm caches outside the counted region.
    for _ in range(args.warmup_iters):
        launch()
    torch.cuda.synchronize()

    print(
        f"profile arm={label} ctx={seq_len} "
        f"window={window} blocks={case.num_blocks_per_seq} "
        f"start_block={case.start_block} "
        f"loop_iters_per_program={case.num_blocks_per_seq - case.start_block} "
        f"grid={(case.num_seqs, case.num_kv_heads)} "
        f"dispatches={args.profile_iters}"
    )
    for _ in range(args.profile_iters):
        launch()
    torch.cuda.synchronize()
    print("profile run complete")
    return 0


def main() -> int:
    parser = FlexibleArgumentParser(
        description="Benchmark the sliding-window block skip in Triton paged decode."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 2048, 4096, 8192, 16384, 32768],
        help="Context lengths to sweep; spans the window-bounded regime change.",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        nargs="+",
        default=[512, 4096],
        help="Windows to sweep. 512 is what real SWA models use.",
    )
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25, help="do_bench warmup ms")
    parser.add_argument("--iters", type=int, default=200, help="do_bench rep ms")
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Alternate the two arms this many times and take the per-arm "
        "median, so clock drift cannot be charged to one arm.",
    )
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Skip the correctness gates (not recommended).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help="If >0, run a single arm this many times for an external profiler.",
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument(
        "--arm",
        type=str,
        choices=["before", "after", "prechange", "production"],
        default="after",
        help="Which kernel to profile: 'before'/'after' are the parametrised "
        "copy's two arms; 'prechange'/'production' are the real kernel from "
        "either side of the change.",
    )
    parser.add_argument(
        "--no-time-registers",
        dest="time_registers",
        action="store_false",
        help="With --report-registers, skip the head-to-head timing of the "
        "two real kernels and report register counts only.",
    )
    parser.add_argument(
        "--report-registers",
        action="store_true",
        help="Report VGPR allocation for the pre- and post-change production "
        "kernels at each --sliding-window value, then exit. Use "
        "--sliding-window 0 512 to check whether a non-windowed layer pays "
        "for the constexpr guard.",
    )
    args = parser.parse_args()

    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        # A skip, not a crash: this file must be importable and runnable as a
        # no-op on a machine with no GPU backend.
        print(
            "skipped: the Triton paged-decode kernel needs a CUDA or ROCm device",
            file=sys.stderr,
        )
        return 0
    if not torch.cuda.is_available():
        print("skipped: no GPU visible to torch", file=sys.stderr)
        return 0
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    if args.report_registers:
        return run_registers(args)
    if args.profile_iters > 0:
        return run_profile(args)
    return run_sweep(args)


if __name__ == "__main__":
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    raise SystemExit(main())
