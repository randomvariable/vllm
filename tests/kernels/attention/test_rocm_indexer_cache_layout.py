# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip proof of the sparse-MLA indexer KV-cache byte layout.

``indexer_k_quant_and_cache_triton`` writes the indexer cache in a shuffled
16x16 tiled layout whenever ``block_size > 1``. Two read paths (the AITER
stage-1 kernel and the Torch reference) interpret those bytes as row-major and
now refuse to run instead of returning silently wrong top-k indices.

Restoring that capability needs a shuffle-aware reader, and writing one against
a *guessed* layout would only manufacture a new silent wrongness. These tests
establish the layout as fact first: a reference unpacker derived independently
from the writer's own index arithmetic, validated by pushing distinctive data
through the production writer on real hardware and recovering it exactly.

The unpacker deliberately does **not** call ``fp8_paged_mqa_logits_torch``.
That function is one of the paths proven unable to read this layout, so using
it as an oracle would be circular.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)

# Production indexer geometry: DeepSeek-V4's indexer cache carries head_dim=128
# FP8 key bytes plus head_dim // quant_block_size * 4 == 4 bytes of FP32 scale.
HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128
SCALE_BYTES = 4
CACHE_LAST_DIM = HEAD_DIM + SCALE_BYTES

# Tile geometry hardcoded by indexer_k_quant_and_cache_triton's defaults.
BLOCK_TILE_SIZE = 16
HEAD_TILE_SIZE = 16

# The writer divides each row's amax by the FP8 flavour's max magnitude to get
# the scale. Feeding rows whose amax *is* that magnitude makes the scale exactly
# the value we chose, with no rounding to reason about.
_FP8_MAX_BY_DTYPE = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e4m3fnuz: 224.0,
}


def _reference_unpack_indexer_cache(
    cache: torch.Tensor,
    page: int,
    token_in_page: int,
    *,
    fp8_dtype: torch.dtype,
    head_dim: int = HEAD_DIM,
    block_tile_size: int = BLOCK_TILE_SIZE,
    head_tile_size: int = HEAD_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover one token's FP8 key row and FP32 scale from the packed cache.

    Written for clarity, not speed, and derived only from the index arithmetic
    in ``_indexer_k_quant_and_cache_kernel``. Within one flattened physical
    page the storage is not interleaved per token; it is a value region
    followed by a scale region::

        [block_size * head_dim FP8 bytes][block_size FP32 scales]

    For ``block_size == 1`` the value region is plain row-major (the writer's
    ``NORMAL`` layout). For ``block_size > 1`` it is shuffled 16x16
    (``SHUFFLE``), i.e. logically indexed as
    ``[page_token_tile, dim_tile, token_lane, dim_lane]``.

    Args:
        cache: Raw indexer cache, shape ``[num_blocks, block_size, head_dim+4]``
            (or that shape with a trailing singleton head dim), dtype uint8.
        page: Physical page index, i.e. the first dimension of ``cache``.
        token_in_page: Token offset within the page, in ``[0, block_size)``.
        fp8_dtype: FP8 encoding the writer stored values in.
        head_dim: Number of FP8 key values per token.
        block_tile_size: Tokens per shuffle tile.
        head_tile_size: Dims per shuffle tile.

    Returns:
        Tuple of ``(values, scale)``: the token's ``head_dim`` FP8 values
        decoded to float32, and its scalar FP32 dequant scale.
    """
    assert cache.dtype == torch.uint8, f"expected a uint8 cache, got {cache.dtype}"
    num_blocks = cache.shape[0]
    block_size = cache.shape[1]
    assert 0 <= page < num_blocks, f"page {page} outside [0, {num_blocks})"
    assert 0 <= token_in_page < block_size, (
        f"token_in_page {token_in_page} outside [0, {block_size})"
    )

    flat_page = cache.reshape(num_blocks, -1)[page]
    value_region = flat_page[: block_size * head_dim]
    scale_region = flat_page[block_size * head_dim :].view(torch.float32)

    dims = torch.arange(head_dim, device=cache.device)
    if block_size == 1:
        byte_offsets = dims
    else:
        token_tile = token_in_page // block_tile_size
        token_lane = token_in_page % block_tile_size
        dim_tile = dims // head_tile_size
        dim_lane = dims % head_tile_size
        byte_offsets = (
            token_tile * block_tile_size * head_dim
            + dim_tile * block_tile_size * head_tile_size
            + token_lane * head_tile_size
            + dim_lane
        )

    values = value_region[byte_offsets].view(fp8_dtype).to(torch.float32)
    return values, scale_region[token_in_page]


def _row_major_unpack_indexer_cache(
    cache: torch.Tensor,
    page: int,
    token_in_page: int,
    *,
    fp8_dtype: torch.dtype,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    """Read a token row assuming the value region is row-major.

    This is what the rejected read paths effectively do. Kept here so a test
    can demonstrate the two interpretations genuinely disagree.
    """
    num_blocks = cache.shape[0]
    block_size = cache.shape[1]
    flat_page = cache.reshape(num_blocks, -1)[page]
    value_region = flat_page[: block_size * head_dim]
    start = token_in_page * head_dim
    row = value_region[start : start + head_dim]
    return row.view(fp8_dtype).to(torch.float32)


def _raw_value_bytes(
    cache: torch.Tensor,
    page: int,
    token_in_page: int,
    *,
    head_dim: int = HEAD_DIM,
    block_tile_size: int = BLOCK_TILE_SIZE,
    head_tile_size: int = HEAD_TILE_SIZE,
) -> torch.Tensor:
    """Return one token's raw value bytes at its layout-correct offsets.

    Used to assert "nothing was written here" without decoding, since whether a
    given byte pattern is NaN depends on the FP8 flavour.
    """
    num_blocks = cache.shape[0]
    block_size = cache.shape[1]
    flat_page = cache.reshape(num_blocks, -1)[page]
    value_region = flat_page[: block_size * head_dim]

    dims = torch.arange(head_dim, device=cache.device)
    if block_size == 1:
        byte_offsets = dims
    else:
        byte_offsets = (
            (token_in_page // block_tile_size) * block_tile_size * head_dim
            + (dims // head_tile_size) * block_tile_size * head_tile_size
            + (token_in_page % block_tile_size) * head_tile_size
            + dims % head_tile_size
        )
    return value_region[byte_offsets]


def _fp8_max(fp8_dtype: torch.dtype) -> float:
    assert fp8_dtype in _FP8_MAX_BY_DTYPE, f"unhandled FP8 dtype {fp8_dtype}"
    return _FP8_MAX_BY_DTYPE[fp8_dtype]


def _exactly_representable_pool(fp8_dtype: torch.dtype, size: int) -> torch.Tensor:
    """Build ``size`` distinct values that survive an FP8 round trip exactly.

    Every entry is decoded straight from a distinct FP8 bit pattern, so casting
    it back to ``fp8_dtype`` cannot round. The flavour's max magnitude is placed
    first so it appears in every rotated row and pins that row's amax.
    """
    fp8_max = _fp8_max(fp8_dtype)
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(fp8_dtype).to(torch.float32)
    values = values[torch.isfinite(values) & (values != 0)]
    values = values[values.abs() != fp8_max]
    # Prefer the largest magnitudes: they are the coarsest-spaced FP8 values, so
    # the products fed to the writer stay exactly representable in bfloat16.
    order = torch.argsort(values.abs(), descending=True)
    chosen = values[order][: size - 1]
    assert chosen.numel() == size - 1, (
        f"only {chosen.numel() + 1} usable {fp8_dtype} values for size {size}"
    )
    pool = torch.cat([torch.tensor([fp8_max]), chosen])
    assert pool.unique().numel() == size, "pool must be free of duplicates"
    return pool


def _token_scale_exponent(token: int) -> int:
    """Per-token power-of-two scale exponent, cycling over 0.25x .. 4x."""
    return (token % 5) - 2


def _build_distinctive_keys(
    num_tokens: int, dtype: torch.dtype, fp8_dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct keys whose exact FP8 quantization is known in advance.

    Each token's row is a rotation of a pool of distinct exactly-representable
    FP8 values, so any swapped dim within a row, or any two rows exchanged
    within a page, changes the recovered vector. ``E4M3FN_MAX`` sits in every
    row, which forces the writer's ``scale`` onto an exact power of two and
    makes ``value / scale`` land back on the pool entry bit-for-bit.

    Returns:
        Tuple of ``(keys, expected_values, expected_scales)`` where ``keys`` is
        the writer input and the other two are the exact FP8 values and scales
        the writer must produce.
    """
    pool = _exactly_representable_pool(fp8_dtype, HEAD_DIM)
    dims = torch.arange(HEAD_DIM)

    expected_values = torch.empty((num_tokens, HEAD_DIM), dtype=torch.float32)
    expected_scales = torch.empty(num_tokens, dtype=torch.float32)
    for token in range(num_tokens):
        # Stride 7 is coprime with HEAD_DIM, so consecutive tokens get
        # genuinely different rotations rather than a shifted duplicate.
        expected_values[token] = pool[(dims + 7 * token) % HEAD_DIM]
        expected_scales[token] = float(2.0 ** _token_scale_exponent(token))

    keys = expected_values * expected_scales[:, None]
    assert torch.equal(keys.abs().amax(dim=1), expected_scales * _fp8_max(fp8_dtype)), (
        "every row must peak at the FP8 max times its scale, or the writer's "
        "amax/max division will not reproduce the scale exactly"
    )
    assert torch.equal(keys.to(dtype).to(torch.float32), keys), (
        f"keys must be exactly representable in {dtype}"
    )
    return (
        keys.to(device=device, dtype=dtype),
        expected_values.to(device),
        expected_scales.to(device),
    )


def _allocate_cache(num_blocks: int, block_size: int, device: str) -> torch.Tensor:
    """Allocate an indexer cache pre-filled with a non-zero sentinel.

    0xFF lets a test tell "the writer wrote this" apart from "this was never
    touched", which zero-fill cannot.
    """
    return torch.full(
        (num_blocks, block_size, CACHE_LAST_DIM),
        0xFF,
        dtype=torch.uint8,
        device=device,
    )


def _slot_mapping_for_pages(
    pages: list[int], num_tokens: int, block_size: int, device: str
) -> torch.Tensor:
    """Map logical token i onto ``pages[i // block_size]``.

    Passing a non-contiguous, out-of-order ``pages`` list is what exercises
    page indexing rather than accidentally validating a linear walk.
    """
    slots = [
        pages[i // block_size] * block_size + (i % block_size)
        for i in range(num_tokens)
    ]
    return torch.tensor(slots, dtype=torch.int64, device=device)


def _write_through_production_kernel(
    keys: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale_fmt: str | None,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    indexer_k_quant_and_cache_triton(
        keys,
        cache,
        slot_mapping,
        QUANT_BLOCK_SIZE,
        scale_fmt,
    )


def _assert_roundtrip(
    cache: torch.Tensor,
    pages: list[int],
    block_size: int,
    expected_values: torch.Tensor,
    expected_scales: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> None:
    num_tokens = expected_values.shape[0]
    for token in range(num_tokens):
        page = pages[token // block_size]
        token_in_page = token % block_size
        values, scale = _reference_unpack_indexer_cache(
            cache, page, token_in_page, fp8_dtype=fp8_dtype
        )
        assert torch.equal(scale, expected_scales[token]), (
            f"scale mismatch for token {token} at page {page}"
            f" offset {token_in_page}: got {scale}, want {expected_scales[token]}"
        )
        assert torch.equal(values, expected_values[token]), (
            f"value mismatch for token {token} at page {page}"
            f" offset {token_in_page}: first differing dim"
            f" {int((values != expected_values[token]).nonzero()[0])}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
def test_indexer_cache_uses_ocp_e4m3fn_off_mi300() -> None:
    """Pin the FP8 encoding the layout proof depends on.

    gfx942-class parts use e4m3fnuz; gfx1151 uses OCP e4m3fn. The reference
    unpacker decodes raw bytes, so reading them with the wrong encoding would
    silently rescale every value by a factor of two.
    """
    from vllm.platforms.rocm import _GCN_ARCH

    fp8_dtype = current_platform.fp8_dtype()
    if "gfx94" in _GCN_ARCH:
        assert fp8_dtype == torch.float8_e4m3fnuz
    else:
        assert fp8_dtype == torch.float8_e4m3fn, (
            f"expected OCP e4m3fn on {_GCN_ARCH}, got {fp8_dtype}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@torch.inference_mode()
def test_normal_layout_roundtrip(dtype: torch.dtype, scale_fmt: str | None) -> None:
    """block_size == 1 degenerates to [head_dim FP8][one FP32] per page."""
    fp8_dtype = current_platform.fp8_dtype()
    block_size = 1
    num_tokens = 5
    # Out-of-order and sparse: page 0 and 2 are deliberately skipped.
    pages = [4, 1, 6, 3, 5]
    cache = _allocate_cache(8, block_size, "cuda")
    keys, expected_values, expected_scales = _build_distinctive_keys(
        num_tokens, dtype, fp8_dtype, "cuda"
    )
    slot_mapping = _slot_mapping_for_pages(pages, num_tokens, block_size, "cuda")

    _write_through_production_kernel(keys, cache, slot_mapping, scale_fmt)

    _assert_roundtrip(
        cache, pages, block_size, expected_values, expected_scales, fp8_dtype
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize(
    "block_size,num_tokens",
    [
        # DSV4's real block size, exactly one full page.
        (64, 64),
        # Partially filled final page, and not a whole multiple of 16 tokens,
        # so the last tile is only fractionally populated.
        (64, 83),
        # Crosses the 16-token tile boundary inside a single page without
        # filling the second tile.
        (64, 20),
        # Smallest shuffled block size: one tile per page.
        (16, 40),
        # Larger page than DSV4 uses, to catch tile-stride assumptions.
        (256, 300),
    ],
)
@torch.inference_mode()
def test_shuffle_layout_roundtrip(
    dtype: torch.dtype, scale_fmt: str | None, block_size: int, num_tokens: int
) -> None:
    """block_size > 1 stores the value region shuffled 16x16.

    This is the case DeepSeek-V4 actually runs, and the one no current read
    path can interpret.
    """
    fp8_dtype = current_platform.fp8_dtype()
    num_pages_used = (num_tokens + block_size - 1) // block_size
    # Non-contiguous, descending page order: nothing here works by accident if
    # page indexing is wrong.
    pages = [2 * i + 1 for i in range(num_pages_used)][::-1]
    cache = _allocate_cache(max(pages) + 2, block_size, "cuda")
    keys, expected_values, expected_scales = _build_distinctive_keys(
        num_tokens, dtype, fp8_dtype, "cuda"
    )
    slot_mapping = _slot_mapping_for_pages(pages, num_tokens, block_size, "cuda")

    _write_through_production_kernel(keys, cache, slot_mapping, scale_fmt)

    _assert_roundtrip(
        cache, pages, block_size, expected_values, expected_scales, fp8_dtype
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
@torch.inference_mode()
def test_shuffle_layout_is_not_row_major() -> None:
    """The shuffled and row-major interpretations genuinely disagree.

    Without this, the round-trip tests above could pass against a layout that
    happens to be row-major anyway, and the guard in
    ``_raise_unsupported_shuffled_layout`` would be pointless.
    """
    fp8_dtype = current_platform.fp8_dtype()
    block_size = 64
    num_tokens = block_size
    pages = [1]
    cache = _allocate_cache(2, block_size, "cuda")
    keys, expected_values, _ = _build_distinctive_keys(
        num_tokens, torch.bfloat16, fp8_dtype, "cuda"
    )
    slot_mapping = _slot_mapping_for_pages(pages, num_tokens, block_size, "cuda")

    _write_through_production_kernel(keys, cache, slot_mapping, "ue8m0")

    # Token 0 dim 0 sits at byte 0 under both readings, so compare a token
    # whose tile/lane decomposition actually moves it.
    token_in_page = 17
    shuffled, _ = _reference_unpack_indexer_cache(
        cache, pages[0], token_in_page, fp8_dtype=fp8_dtype
    )
    row_major = _row_major_unpack_indexer_cache(
        cache, pages[0], token_in_page, fp8_dtype=fp8_dtype
    )

    assert torch.equal(shuffled, expected_values[token_in_page])
    assert not torch.equal(row_major, expected_values[token_in_page]), (
        "row-major read recovered the right row, so the cache is not shuffled"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
@pytest.mark.parametrize("block_size", [1, 64])
@torch.inference_mode()
def test_writer_leaves_unmapped_pages_untouched(block_size: int) -> None:
    """Writes must stay inside their own page.

    The packed layout makes per-page strides large, and the kernel casts
    ``block_id`` to int64 specifically to avoid 32-bit overflow. A page that
    was never mapped must still hold the sentinel.
    """
    fp8_dtype = current_platform.fp8_dtype()
    num_tokens = block_size * 2
    pages = [3, 0]
    num_blocks = 6
    cache = _allocate_cache(num_blocks, block_size, "cuda")
    keys, _, _ = _build_distinctive_keys(num_tokens, torch.bfloat16, fp8_dtype, "cuda")
    slot_mapping = _slot_mapping_for_pages(pages, num_tokens, block_size, "cuda")

    _write_through_production_kernel(keys, cache, slot_mapping, "ue8m0")

    untouched = [p for p in range(num_blocks) if p not in pages]
    for page in untouched:
        assert torch.all(cache[page] == 0xFF), f"writer spilled into page {page}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
@torch.inference_mode()
def test_negative_slots_are_skipped() -> None:
    """A negative slot means "no cache insert" and must write nothing."""
    fp8_dtype = current_platform.fp8_dtype()
    block_size = 64
    num_tokens = 8
    cache = _allocate_cache(2, block_size, "cuda")
    keys, expected_values, expected_scales = _build_distinctive_keys(
        num_tokens, torch.bfloat16, fp8_dtype, "cuda"
    )
    slot_mapping = torch.tensor(
        [0, -1, 2, -1, 4, -1, 6, -1], dtype=torch.int64, device="cuda"
    )

    _write_through_production_kernel(keys, cache, slot_mapping, "ue8m0")

    for token, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        values, scale = _reference_unpack_indexer_cache(
            cache, slot // block_size, slot % block_size, fp8_dtype=fp8_dtype
        )
        assert torch.equal(values, expected_values[token])
        assert torch.equal(scale, expected_scales[token])

    # Odd offsets were only ever named by negative slots, so their bytes must
    # still hold the fill sentinel. Compared raw: whether a byte pattern reads
    # as NaN differs between e4m3fn and e4m3fnuz.
    for offset in range(1, 8, 2):
        untouched = _raw_value_bytes(cache, 0, offset)
        assert torch.all(untouched == 0xFF), (
            f"negative slot wrote into page 0 offset {offset}"
        )
