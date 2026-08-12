# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)


def _on_split_decode_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950

        return bool(_ON_GFX942 or _ON_GFX950)
    except Exception:
        return False


# The flash-decode split-K decode path is only tuned for AMD gfx942/gfx950; other
# architectures take the fallback decode kernel, so its tests are skipped there.
requires_split_decode_arch = pytest.mark.skipif(
    not _on_split_decode_arch(),
    reason="split-K decode kernel is only tuned for AMD gfx942/gfx950",
)

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM


def _ref_global_topk_ragged(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = topk_indices.reshape(topk_indices.shape[0], -1)
    valid = (topk >= 0) & is_valid_token[:, None]
    lens = valid.sum(dim=1, dtype=torch.int32)
    indptr = torch.zeros(lens.shape[0] + 1, dtype=torch.int32, device=topk.device)
    torch.cumsum(lens, dim=0, out=indptr[1:])

    safe_topk = torch.clamp(topk, min=0)
    block_indices = safe_topk // block_size
    block_offsets = safe_topk % block_size
    req_indices = token_to_req_indices[:, None].expand_as(topk)
    slot_ids = block_table[req_indices, block_indices] * block_size + block_offsets

    offsets = torch.arange(topk.shape[1], dtype=torch.int32, device=topk.device)
    positions = indptr[:-1, None] + offsets[None, :]
    return slot_ids[valid], positions[valid].to(torch.long), indptr, lens


def _ref_sparse_prefill_ragged(
    q: torch.Tensor,
    kv: torch.Tensor,
    rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_indices = rows[query_idx]
        for head_idx in range(q.shape[1]):
            if row_indices:
                selected_kv = kv_f32[row_indices]
                scores = torch.mv(selected_kv, q_f32[query_idx, head_idx]) * scale
                if attn_sink is not None:
                    scores_with_sink = torch.cat(
                        [scores, attn_sink[head_idx].float().reshape(1)]
                    )
                    probs = torch.softmax(scores_with_sink, dim=0)[:-1]
                else:
                    probs = torch.softmax(scores, dim=0)
                out[query_idx, head_idx] = torch.sum(
                    probs[:, None] * selected_kv, dim=0
                )
            else:
                out[query_idx, head_idx] = 0
    return out.to(torch.bfloat16)


def _pack_fp8_ds_mla_cache(
    kv: torch.Tensor, block_size: int, use_fnuz: bool
) -> torch.Tensor:
    assert kv.shape[-1] == HEAD_DIM
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584),
        dtype=torch.uint8,
        device=kv.device,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=kv.device)
    quantize_and_insert_k_cache(
        kv,
        cache,
        slot_mapping,
        block_size=block_size,
        use_fnuz=use_fnuz,
    )
    return cache


def _read_fp8_ds_mla_cache(
    cache: torch.Tensor, slot: int, block_size: int, use_fnuz: bool
) -> torch.Tensor:
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8

    fp8_dtype = torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
    nope_u8 = cache_flat[token_base : token_base + NOPE_HEAD_DIM]
    nope = nope_u8.view(fp8_dtype).to(torch.float32)
    scales = torch.exp2(
        cache_flat[scale_base : scale_base + 7].to(torch.float32) - 127.0
    )
    nope = nope * scales.repeat_interleave(64)
    rope_u8 = cache_flat[
        token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
    ]
    rope = rope_u8.view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope])


def _ref_sparse_decode_ragged(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
    block_size: int,
    extra_cache: torch.Tensor | None = None,
    extra_rows: list[list[int]] | None = None,
    main_use_fnuz: bool = False,
    extra_use_fnuz: bool = False,
) -> torch.Tensor:
    q_f32 = q.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_kv = [
            _read_fp8_ds_mla_cache(main_cache, int(slot), block_size, main_use_fnuz)
            for slot in main_rows[query_idx]
        ]
        if extra_cache is not None and extra_rows is not None:
            row_kv.extend(
                _read_fp8_ds_mla_cache(
                    extra_cache, int(slot), block_size, extra_use_fnuz
                )
                for slot in extra_rows[query_idx]
            )

        kv = torch.stack(row_kv).to(q.device)
        for head_idx in range(q.shape[1]):
            scores = torch.mv(kv, q_f32[query_idx, head_idx]) * scale
            if attn_sink is not None:
                scores_with_sink = torch.cat(
                    [scores, attn_sink[head_idx].float().reshape(1)]
                )
                probs = torch.softmax(scores_with_sink, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[query_idx, head_idx] = torch.sum(probs[:, None] * kv, dim=0)
    return out.to(torch.bfloat16)


def _ragged_from_rows(
    rows: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-query slot lists into ragged (indices, indptr) tensors."""
    flat = [slot for row in rows for slot in row]
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + len(row))
    return (
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
    )


@torch.inference_mode()
def test_paged_mqa_logits_do_not_contain_nan(monkeypatch) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")

    class FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            return [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in shapes_and_dtypes
            ]

    def fake_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_logits,
        context_lens,
        block_tables,
        max_seq_len,
        **kwargs,
    ):
        del (
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_seq_len,
            kwargs,
        )
        out_logits.fill_(float("nan"))

    monkeypatch.setattr(mod, "_ON_GFX942", False)
    monkeypatch.setattr(mod, "_ON_GFX950", True)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(deepgemm_fp8_paged_mqa_logits=fake_paged_mqa_logits),
    )
    monkeypatch.setattr(
        mod, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )

    q_fp8 = torch.empty((1, 1, 1, 1), dtype=torch.uint8, device=device)
    kv_cache_fp8 = torch.empty((1, 1, 1, 5), dtype=torch.uint8, device=device)
    logits = mod.rocm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        torch.empty((1, 1), dtype=torch.float32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        1,
    )

    assert not torch.isnan(logits).any()


@torch.inference_mode()
def test_compute_global_topk_ragged_indices_and_indptr() -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )

    device = torch.device("cuda")
    block_size = 4
    topk_indices = torch.tensor(
        [
            [0, 3, 4, -1],
            [5, 8, -1, -1],
            [2, 7, 9, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.int32,
        device=device,
    )
    is_valid_token = torch.tensor([True, False, True], dtype=torch.bool, device=device)

    actual_ragged, actual_indptr, actual_lens = (
        compute_global_topk_ragged_indices_and_indptr(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
    )
    expected_values, expected_positions, expected_indptr, expected_lens = (
        _ref_global_topk_ragged(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
    )

    torch.testing.assert_close(actual_ragged[expected_positions], expected_values)
    torch.testing.assert_close(actual_indptr, expected_indptr)
    torch.testing.assert_close(actual_lens, expected_lens)


@torch.inference_mode()
def test_sparse_attn_prefill_ragged_kernel() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(3, 3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    indices = torch.tensor([0, 2, 1, 3, 4], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
    )
    expected = _ref_sparse_prefill_ragged(
        q, kv, [[0, 2], [1, 3, 4], []], scale, attn_sink
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_sparse_attn_decode_ragged_kernel() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(1)
    block_size = 4
    main_use_fnuz = current_platform.is_fp8_fnuz()
    q = torch.randn(2, 3, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, use_fnuz=main_use_fnuz)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, use_fnuz=False)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    extra_indices = torch.tensor([1, 3, 0], dtype=torch.int32, device=device)
    extra_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=[[0, 2], [4, 1]],
        scale=scale,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=[[1], [3, 0]],
        main_use_fnuz=main_use_fnuz,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@requires_split_decode_arch
@torch.inference_mode()
def test_decode_num_splits_heuristic(monkeypatch) -> None:
    """Split-count heuristic added with the flash-decode split-K decode path."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    # Pin the CU count so the heuristic is deterministic off-device.
    monkeypatch.setattr(mod, "_decode_cu_count", lambda: 256)

    # A batch that already fills the device should not be split.
    assert mod._decode_num_splits(256, 1, avg_main_len=128.0, avg_extra_len=0.0) == 1
    # A tiny batch on a large device should split to add parallelism.
    assert mod._decode_num_splits(2, 1, avg_main_len=256.0, avg_extra_len=0.0) > 1

    # The chosen count always stays within the searched [1, 16] range, and a
    # zero-length workload never splits (no work to parallelize).
    for num_queries in (1, 4, 24, 224, 1024):
        splits = mod._decode_num_splits(
            num_queries, 1, avg_main_len=512.0, avg_extra_len=128.0
        )
        assert 1 <= splits <= 16
    assert mod._decode_num_splits(2, 1, avg_main_len=0.0, avg_extra_len=0.0) >= 1


@requires_split_decode_arch
@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("with_extra", [True, False])
@pytest.mark.parametrize("with_sink", [True, False])
@torch.inference_mode()
def test_sparse_attn_decode_split_k_kernel(
    monkeypatch, num_splits: int, with_extra: bool, with_sink: bool
) -> None:
    """Flash-decode split-K decode path (partial + reduce kernels).

    This path is the gfx942/gfx950 production path, so the test only runs on
    those architectures. The split count is pinned so the partial/reduce kernels are
    exercised across split counts. ``num_splits=8`` drives splits past the
    shortest segment length, covering the empty-split edge case handled by the
    reduce kernel.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    device = torch.device("cuda")
    torch.manual_seed(7)
    block_size = 4
    num_heads = 3
    main_use_fnuz = current_platform.is_fp8_fnuz()

    main_rows = [[0, 2, 4, 6, 1, 3, 7, 5], [4, 1, 6, 0, 2]]
    num_queries = len(main_rows)
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(8, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, use_fnuz=main_use_fnuz)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)

    extra_rows: list[list[int]] | None = None
    extra_cache: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_indptr: torch.Tensor | None = None
    if with_extra:
        rows = [[1, 3, 0, 5, 2, 4], [3, 0, 6]]
        extra_kv = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
        extra_rows = rows
        extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, use_fnuz=False)
        extra_indices, extra_indptr = _ragged_from_rows(rows, device)

    attn_sink = (
        torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    # Pin the split count so each parametrized value is exercised deterministically.
    monkeypatch.setattr(mod, "_decode_num_splits", lambda *args, **kwargs: num_splits)

    actual = mod._rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    expected = _ref_sparse_decode_ragged(
        q=q,
        main_cache=main_cache,
        main_rows=main_rows,
        scale=scale,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
        main_use_fnuz=main_use_fnuz,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# o-projection: fused inverse-RoPE + cached bf16 wo_a (rocm_inv_rope_einsum)
# ---------------------------------------------------------------------------


# Cache rows = max_position_embeddings * scaling_factor.
_ROTARY_MAX_POS = 1024
_ROTARY_SCALING_FACTOR = 4.0
_ROTARY_CACHE_LEN = int(_ROTARY_MAX_POS * _ROTARY_SCALING_FACTOR)


def _make_dsv4_rotary(device: torch.device):
    """The official DSv4 rotary embedding, sized down for unit tests."""
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

    # The model loader constructs layers under a default-device context;
    # mirror that so the fp32 cos_sin_cache lands on the GPU.
    with torch.device(device):
        rotary_emb = DeepseekV4ScalingRotaryEmbedding(
            head_size=ROPE_HEAD_DIM,
            rotary_dim=ROPE_HEAD_DIM,
            max_position_embeddings=_ROTARY_MAX_POS,
            base=10000,
            is_neox_style=False,
            scaling_factor=_ROTARY_SCALING_FACTOR,
            dtype=torch.bfloat16,
            mscale=1.0,
            mscale_all_dim=1.0,
        )
    rotary_emb = rotary_emb.to(device)
    assert rotary_emb.cos_sin_cache.shape == (_ROTARY_CACHE_LEN, ROPE_HEAD_DIM)
    return rotary_emb


def _inv_rope_via_rotary_native(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference: the official ``forward_native(inverse=True)`` path."""
    expected, _ = rotary_emb.forward_native(positions, o.clone(), None, inverse=True)
    return expected.to(torch.bfloat16)


class _FakeWoA(torch.nn.Module):
    """Stand-in for the wo_a linear layer holding the (optionally fp8) weight."""

    def __init__(
        self, weight: torch.Tensor, weight_scale_inv: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.weight = weight
        if weight_scale_inv is not None:
            self.weight_scale_inv = weight_scale_inv


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_fused_inverse_rope_gptj_matches_rotary_native(
    num_tokens: int, num_heads: int, pos_dtype: torch.dtype, default_vllm_config
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

    device = torch.device("cuda")
    torch.manual_seed(0)
    rotary_emb = _make_dsv4_rotary(device)
    o = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, _ROTARY_CACHE_LEN, (num_tokens,), dtype=pos_dtype, device=device
    )

    actual = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, ROPE_HEAD_DIM
    )
    expected = _inv_rope_via_rotary_native(rotary_emb, o, positions)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == o.shape
    # NoPE lanes are a pure bf16 passthrough -> must be bit-exact.
    assert torch.equal(actual[..., :NOPE_HEAD_DIM], expected[..., :NOPE_HEAD_DIM])
    # RoPE lanes: tolerate at most ~1 bf16 ulp from fp32 fma ordering.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_fused_inverse_rope_gptj_empty(default_vllm_config) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

    device = torch.device("cuda")
    rotary_emb = _make_dsv4_rotary(device)
    o = torch.empty(0, 8, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.empty(0, dtype=torch.int32, device=device)

    out = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, ROPE_HEAD_DIM
    )
    assert out.shape == (0, 8, HEAD_DIM)
    assert out.dtype == torch.bfloat16


@torch.inference_mode()
def test_rocm_inv_rope_einsum_matches_rotary_native(default_vllm_config) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum

    device = torch.device("cuda")
    torch.manual_seed(2)
    num_tokens, num_heads = 5, 8
    n_local_groups = num_heads
    o_lora_rank = 16
    hidden_dim = num_heads * HEAD_DIM // n_local_groups  # 512

    rotary_emb = _make_dsv4_rotary(device)
    o = (
        torch.randn(
            num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    positions = torch.randint(
        0, _ROTARY_CACHE_LEN, (num_tokens,), dtype=torch.int32, device=device
    )
    weight = (
        torch.randn(n_local_groups * o_lora_rank, hidden_dim, device=device) * 0.125
    ).to(torch.bfloat16)
    wo_a = _FakeWoA(weight)

    actual = rocm_inv_rope_einsum(
        rotary_emb, o, positions, ROPE_HEAD_DIM, n_local_groups, o_lora_rank, wo_a
    )

    o_ref = _inv_rope_via_rotary_native(rotary_emb, o, positions)
    o_ref = o_ref.view(num_tokens, n_local_groups, -1)
    wo_a_ref = weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    expected = torch.einsum("tgd,grd->tgr", o_ref, wo_a_ref)

    assert actual.shape == (num_tokens, n_local_groups, o_lora_rank)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_get_cached_wo_a_bf16_plain_caches() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    device = torch.device("cuda")
    torch.manual_seed(4)
    n_local_groups, o_lora_rank, hidden_dim = 2, 4, 8
    weight = torch.randn(
        n_local_groups * o_lora_rank, hidden_dim, dtype=torch.bfloat16, device=device
    )
    wo_a = _FakeWoA(weight)

    out1 = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)
    expected = weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    assert out1.shape == (n_local_groups, o_lora_rank, hidden_dim)
    torch.testing.assert_close(out1, expected, atol=0, rtol=0)
    assert hasattr(wo_a, "_dsv4_wo_a_bf16")

    # Mutate the source weight: the cached tensor must be returned unchanged
    # (proving the dequant is not recomputed per call).
    wo_a.weight.zero_()
    out2 = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)
    assert out2 is out1
    torch.testing.assert_close(out2, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_get_cached_wo_a_bf16_fp8_blockscale_caches() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    device = torch.device("cuda")
    torch.manual_seed(5)
    n_local_groups, o_lora_rank, hidden_dim = 2, 4, 8
    row_block, col_block = 2, 2
    row_blocks = o_lora_rank // row_block
    col_blocks = hidden_dim // col_block

    fp8_dtype = current_platform.fp8_dtype()
    weight_f32 = (
        torch.randn(
            n_local_groups, o_lora_rank, hidden_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )
    weight_fp8 = weight_f32.to(fp8_dtype)
    scale = (
        torch.rand(
            n_local_groups, row_blocks, col_blocks, dtype=torch.float32, device=device
        )
        * 0.5
        + 0.5
    )
    wo_a = _FakeWoA(
        weight_fp8.reshape(n_local_groups * o_lora_rank, hidden_dim),
        weight_scale_inv=scale.reshape(n_local_groups * row_blocks, col_blocks),
    )

    out = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim)

    scale_full = scale.repeat_interleave(row_block, dim=-2).repeat_interleave(
        col_block, dim=-1
    )
    expected = (weight_fp8.to(torch.float32) * scale_full).to(torch.bfloat16)
    assert out.shape == (n_local_groups, o_lora_rank, hidden_dim)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    # Second call returns the same cached object.
    assert _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim) is out


# --------------------------------------------------------------------------
# Sparse-MLA indexer dispatch observability.
#
# These exercise the dispatch bookkeeping in ``rocm_fp8_paged_mqa_logits``
# (import-failure diagnostic, route diagnostic, unsupported-layout guard),
# not any kernel, so they mock architecture and AITER availability and run
# on CPU without AITER installed.
# --------------------------------------------------------------------------


def _cpu_paged_mqa_logits_args(
    block_size: int, *, next_n: int = 1, heads: int = 1
) -> tuple:
    """Minimal CPU tensors shaped so dispatch reaches the route decision.

    ``head_dim`` is 4 so the packed ``[block_size, head_dim + 4]`` cache keeps
    the fp32 scale section 4-byte aligned; the Torch reference reinterprets
    that section and would otherwise fail on alignment rather than on the
    behaviour under test.
    """
    head_dim = 4
    return (
        torch.zeros((1, next_n, heads, head_dim), dtype=torch.uint8),  # q_fp8
        # [num_blocks, block_size, 1, head_dim + 4]
        torch.zeros((1, block_size, 1, head_dim + 4), dtype=torch.uint8),
        torch.zeros((next_n, heads), dtype=torch.float32),  # weights
        torch.ones(1, dtype=torch.int32),  # context_lens
        torch.zeros((1, 1), dtype=torch.int32),  # block_tables
        torch.zeros(0, dtype=torch.int32),  # schedule_metadata
        block_size,  # max_model_len, must cover one full block
    )


def _stub_paged_logits_dispatch(
    monkeypatch,
    mod,
    *,
    on_mi3xx: bool,
    aiter_module,
) -> None:
    """Pin arch and AITER availability, and stub out the kernels/workspace."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(mod, "_ON_GFX942", on_mi3xx)
    monkeypatch.setattr(mod, "_ON_GFX950", False)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(mod, "paged_mqa_logits_module", lambda: aiter_module)

    class _FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            return [
                torch.zeros(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes
            ]

    monkeypatch.setattr(
        mod, "current_workspace_manager", lambda: _FakeWorkspaceManager()
    )


def _fake_fused_kernel(*args, **kwargs) -> None:
    """Stand-in for the fused AITER kernel; records nothing, writes nothing."""
    return None


def _fake_stage1_kernel(*args, **kwargs) -> None:
    return None


def test_paged_mqa_logits_aiter_non_mi3xx_uses_torch_for_shuffle(monkeypatch):
    """Non-MI300 AITER dispatch uses Torch for single-token shuffle decode."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    aiter_module = SimpleNamespace(
        deepgemm_fp8_paged_mqa_logits=_fake_fused_kernel,
        deepgemm_fp8_paged_mqa_logits_stage1=lambda *args, **kwargs: pytest.fail(
            "stage-1 must not handle shuffled decode"
        ),
    )
    _stub_paged_logits_dispatch(
        monkeypatch, mod, on_mi3xx=False, aiter_module=aiter_module
    )
    monkeypatch.setattr(
        mod,
        "fp8_paged_mqa_logits_torch",
        lambda *args: torch.zeros((1, 64), dtype=torch.float32),
    )
    reported: list[tuple[str, int]] = []
    monkeypatch.setattr(
        mod,
        "_report_paged_logits_route",
        lambda route, block_size: reported.append((route, block_size)),
    )

    mod.rocm_fp8_paged_mqa_logits(
        *_cpu_paged_mqa_logits_args(block_size=64, next_n=1)
    )

    assert reported == [(mod._PAGED_LOGITS_ROUTE_TORCH, 64)]


def test_paged_mqa_logits_module_import_failure_is_logged_once(monkeypatch):
    """A failed AITER import must report the underlying reason, once.

    Silently returning None is how a deployment lands on the slow Torch
    reference for a reason nobody can recover from the logs.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(
        mod,
        "find_spec",
        lambda name: object() if name == "aiter.ops.triton.pa_mqa_logits" else None,
    )

    boom = ImportError("libaiter.so: cannot open shared object file")

    def _raise_import_error(name):
        raise boom

    monkeypatch.setattr(mod.importlib, "import_module", _raise_import_error)

    logged: list[tuple] = []
    monkeypatch.setattr(
        mod.logger,
        "warning_once",
        lambda msg, *args, **kwargs: logged.append((msg, args)),
    )

    # Bypass the lru_cache so the diagnostic is observable in-test.
    assert mod.paged_mqa_logits_module.__wrapped__() is None

    assert len(logged) == 1, "import failure must be reported exactly once"
    msg, args = logged[0]
    assert "Failed to import AITER paged-MQA logits module" in msg
    # The actual exception must be recoverable from the log, not just the fact
    # that something failed.
    assert boom in args


def test_paged_mqa_logits_module_is_cached_so_diagnostic_fires_once(monkeypatch):
    """The resolver is lru_cached, so a repeated miss logs only on first call."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "find_spec", lambda name: None)

    calls: list[str] = []
    monkeypatch.setattr(
        mod.logger, "warning_once", lambda msg, *a, **k: calls.append(msg)
    )

    cached = functools.lru_cache(mod.paged_mqa_logits_module.__wrapped__)
    assert cached() is None
    assert cached() is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "on_mi3xx,aiter_available,expected_route",
    [
        (True, True, "_PAGED_LOGITS_ROUTE_FUSED"),
        (False, True, "_PAGED_LOGITS_ROUTE_STAGE1"),
        (False, False, "_PAGED_LOGITS_ROUTE_TORCH"),
        (True, False, "_PAGED_LOGITS_ROUTE_TORCH"),
    ],
)
@torch.inference_mode()
def test_paged_mqa_logits_reports_selected_route(
    monkeypatch, on_mi3xx, aiter_available, expected_route
):
    """Each reachable route names itself, so logs distinguish the paths.

    Without AITER the arch is irrelevant: both gfx942 and gfx1151 fall back to
    the Torch reference.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    aiter_module = (
        SimpleNamespace(
            deepgemm_fp8_paged_mqa_logits=_fake_fused_kernel,
            deepgemm_fp8_paged_mqa_logits_stage1=_fake_stage1_kernel,
        )
        if aiter_available
        else None
    )
    _stub_paged_logits_dispatch(
        monkeypatch, mod, on_mi3xx=on_mi3xx, aiter_module=aiter_module
    )

    reported: list[tuple[str, int]] = []
    monkeypatch.setattr(
        mod,
        "_report_paged_logits_route",
        lambda route, block_size: reported.append((route, block_size)),
    )

    # block_size 1 is the supported layout on every route.
    mod.rocm_fp8_paged_mqa_logits(*_cpu_paged_mqa_logits_args(block_size=1))

    assert reported == [(getattr(mod, expected_route), 1)]


def test_report_paged_logits_route_logs_once_with_layout(monkeypatch):
    """The route diagnostic is deduplicated and names arch, block size, layout.

    Decode calls this per request, so it must not log per request.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    logged: list[tuple] = []
    monkeypatch.setattr(
        mod.logger, "info_once", lambda msg, *args, **kwargs: logged.append(args)
    )
    monkeypatch.setattr(mod, "_GCN_ARCH", "gfx1151")

    mod._report_paged_logits_route(mod._PAGED_LOGITS_ROUTE_TORCH, 1)

    assert logged == [(mod._PAGED_LOGITS_ROUTE_TORCH, "gfx1151", 1, "NORMAL")]

    logged.clear()
    mod._report_paged_logits_route(mod._PAGED_LOGITS_ROUTE_FUSED, 64)
    assert logged == [(mod._PAGED_LOGITS_ROUTE_FUSED, "gfx1151", 64, "SHUFFLE")]


@pytest.mark.parametrize("block_size", [16, 64, 256])
@torch.inference_mode()
def test_paged_mqa_logits_torch_handles_shuffled_layout_off_fused_path(
    monkeypatch, block_size
):
    """Non-fused shuffled layouts use the Torch implementation."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    aiter_module = SimpleNamespace(
        deepgemm_fp8_paged_mqa_logits=_fake_fused_kernel,
        deepgemm_fp8_paged_mqa_logits_stage1=_fake_stage1_kernel,
    )
    # on_mi3xx=False keeps us off the fused path in both cases.
    _stub_paged_logits_dispatch(
        monkeypatch, mod, on_mi3xx=False, aiter_module=aiter_module
    )

    called = []
    monkeypatch.setattr(
        mod,
        "fp8_paged_mqa_logits_torch",
        lambda *args: called.append(args) or torch.empty(0),
    )
    args = _cpu_paged_mqa_logits_args(block_size=block_size, next_n=2)
    result = mod.rocm_fp8_paged_mqa_logits(*args)
    assert called
    assert result.numel() == 0


@torch.inference_mode()
def test_paged_mqa_logits_torch_reads_shuffle_layout(monkeypatch):
    """Torch fallback decodes shuffled values across token and dim tiles."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    fp8_dtype = torch.float8_e4m3fn
    monkeypatch.setattr(mod.current_platform, "fp8_dtype", lambda: fp8_dtype)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(mod, "_ON_GFX942", False)
    monkeypatch.setattr(mod, "_ON_GFX950", False)

    block_size, dim, heads = 64, 128, 3
    logical_values = torch.tensor(
        [
            [((token + 1) * (lane + 3) % 29 - 14) / 8 for lane in range(dim)]
            for token in range(block_size)
        ],
        dtype=torch.float32,
    )
    cache = torch.zeros((1, block_size, 1, dim + 4), dtype=torch.uint8)
    flat_page = cache.reshape(1, -1)[0]
    # Build physical bytes independently: writer order is
    # [token_tile, dim_tile, token_lane, dim_lane].
    packed_values = logical_values.reshape(
        block_size // 16, 16, dim // 16, 16
    ).permute(0, 2, 1, 3).contiguous()
    flat_page[: block_size * dim] = packed_values.flatten().to(fp8_dtype).view(
        torch.uint8
    )
    scales = torch.linspace(0.25, 1.75, block_size)
    flat_page[block_size * dim :].view(torch.float32).copy_(scales)

    q = torch.tensor(
        [
            [
                [[((head + 1) * (lane + 1) % 17 - 8) / 4 for lane in range(dim)]
                 for head in range(heads)]
            ]
        ],
        dtype=torch.float32,
    ).to(fp8_dtype)
    weights = torch.tensor([[0.5, 1.25, 2.0]], dtype=torch.float32)
    logits = mod.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        torch.tensor([block_size], dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        block_size,
    )
    quantized_values = logical_values.to(fp8_dtype).to(torch.float32)
    expected = torch.zeros(block_size, dtype=torch.float32)
    for token in range(block_size):
        for head in range(heads):
            dot = torch.dot(
                q[0, 0, head].to(torch.float32), quantized_values[token]
            )
            expected[token] = expected[token] + torch.relu(dot) * weights[0, head]
        expected[token] *= scales[token]
    torch.testing.assert_close(logits[0, :block_size], expected)


@torch.inference_mode()
def test_paged_mqa_logits_torch_reads_multi_token_shuffle_layout(monkeypatch):
    """Torch fallback reads independently packed speculative shuffled pages."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    fp8_dtype = torch.float8_e4m3fn
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(mod, "_ON_GFX942", False)
    monkeypatch.setattr(mod, "_ON_GFX950", False)

    monkeypatch.setattr(mod.current_platform, "fp8_dtype", lambda: fp8_dtype)

    block_size, dim, heads = 64, 128, 2
    batch_size, next_n, max_model_len = 2, 3, 64
    context_lens = torch.tensor([[17, 33, 64], [16, 31, 48]], dtype=torch.int32)
    block_tables = torch.tensor([[4, 1], [7, 3]], dtype=torch.int32)
    num_pages = 8

    cache = torch.zeros((num_pages, block_size, 1, dim + 4), dtype=torch.uint8)
    page_values: dict[int, torch.Tensor] = {}
    for page in block_tables.flatten().unique().tolist():
        values = torch.tensor(
            [
                [((page + 1) * (token + 3) * (lane + 5) % 37 - 18) / 7
                 for lane in range(dim)]
                for token in range(block_size)
            ],
            dtype=torch.float32,
        )
        scales = torch.linspace(0.25 + page / 20, 1.5 + page / 20, block_size)
        packed_values = values.reshape(4, 16, 8, 16).permute(0, 2, 1, 3).contiguous()
        flat_page = cache[page, :, 0].reshape(-1)
        flat_page[: block_size * dim] = packed_values.flatten().to(fp8_dtype).view(
            torch.uint8
        )
        flat_page[block_size * dim :].view(torch.float32).copy_(scales)
        page_values[page] = values.to(fp8_dtype).to(torch.float32) * scales[:, None]

    q_values = torch.tensor(
        [
            [
                [
                    [((query + 2) * (token + 1) * (head + 3) * (lane + 1) % 31 - 15)
                     / 11
                     for lane in range(dim)]
                    for head in range(heads)
                ]
                for token in range(next_n)
            ]
            for query in range(batch_size)
        ],
        dtype=torch.float32,
    )
    q = q_values.to(fp8_dtype).view(batch_size, next_n, heads, dim)
    weights = torch.tensor(
        [[0.5, 1.25], [0.75, 1.5], [1.0, 1.75], [0.6, 1.1], [0.9, 1.3], [1.2, 1.6]],
        dtype=torch.float32,
    )

    actual = mod.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0, dtype=torch.int32),
        max_model_len,
    )

    expected = torch.full(
        (batch_size * next_n, max_model_len), float("-inf"), dtype=torch.float32
    )
    for batch_idx in range(batch_size):
        for token_idx in range(next_n):
            row = batch_idx * next_n + token_idx
            context_len = int(context_lens[batch_idx, token_idx])
            query = q[row // next_n, token_idx].float()
            for position in range(context_len):
                page_idx = int(block_tables[batch_idx, position // block_size])
                key = page_values[page_idx][position % block_size]
                expected[row, position] = sum(
                    torch.relu(torch.dot(query[head], key)) * weights[row, head]
                    for head in range(heads)
                )

    torch.testing.assert_close(actual, expected)
    for row, context_len in enumerate(context_lens.flatten().tolist()):
        expected_topk = torch.argsort(expected[row, :context_len], descending=True)[:8]
        actual_topk = torch.argsort(actual[row, :context_len], descending=True)[:8]
        torch.testing.assert_close(actual_topk, expected_topk)


@pytest.mark.parametrize("block_size", [1, 64, 256])
@torch.inference_mode()
def test_paged_mqa_logits_fused_path_accepts_any_block_size(monkeypatch, block_size):
    """The fused gfx942/gfx950 kernel handles packing, so it stays ungated.

    It receives Preshuffle/KVBlockSize, so gating it would break a correct
    path.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    seen: list[dict] = []

    def _record_fused(*args, **kwargs):
        seen.append(kwargs)

    _stub_paged_logits_dispatch(
        monkeypatch,
        mod,
        on_mi3xx=True,
        aiter_module=SimpleNamespace(
            deepgemm_fp8_paged_mqa_logits=_record_fused,
            deepgemm_fp8_paged_mqa_logits_stage1=_fake_stage1_kernel,
        ),
    )

    mod.rocm_fp8_paged_mqa_logits(*_cpu_paged_mqa_logits_args(block_size=block_size))

    assert len(seen) == 1
    # The layout is communicated to the kernel, which is why it is safe.
    assert seen[0]["KVBlockSize"] == block_size
    assert seen[0]["Preshuffle"] is (block_size > 1)


def test_deepseek_v4_indexer_cache_block_size_is_64():
    """Pin the reachable indexer cache block size for DeepSeek-V4 on ROCm.

    ``DeepseekV4IndexerBackend`` advertises only [256], and the indexer cache
    spec carries compress_ratio=4, so the KV tensor is built with
    ``storage_block_size`` = 256 // 4 = 64. block_size > 1 is therefore the
    normal case, not a hypothetical, which is what makes the guard load-bearing.
    """
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [256]

    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        compress_ratio=4,
        alignment=512,
    )
    assert spec.storage_block_size == 64

    # initialize_kv_cache_tensors uses storage_block_size when it diverges
    # from block_size, so this is the block dim the kernels actually see.
    assert spec.storage_block_size != spec.block_size
    shape = DeepseekV4IndexerBackend.get_kv_cache_shape(
        4, spec.storage_block_size, 1, 132
    )
    assert shape[1] == 64
