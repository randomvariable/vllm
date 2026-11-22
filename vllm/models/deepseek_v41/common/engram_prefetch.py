# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side Engram row prediction for advisory disk prefetch."""

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import torch

from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
from vllm.utils.b12x import get_b12x_engram, get_b12x_engram_host
from vllm.utils.b12x_row_prefetch import PrefetchTarget

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.common.engram import Engram, NgramHashState


def engram_prefetch_targets(
    hash_state: "NgramHashState", layers: Sequence["Engram"]
) -> tuple[PrefetchTarget, ...]:
    layout = hash_state.layout
    native = get_b12x_engram()
    host = get_b12x_engram_host()
    if (
        native is None
        or host is None
        or (layout.max_ngram_size, layout.n_heads) != (4, 8)
    ):
        return ()

    token_map = hash_state.token_map.to(device="cpu", dtype=torch.int64)
    geometry = native.Geometry(
        layer_ids=layout.layer_ids,
        primes=tuple(tuple(row) for row in hash_state.primes.cpu().flatten(1).tolist()),
        offsets=tuple(tuple(row) for row in hash_state.offsets.cpu().tolist()),
        multipliers=tuple(tuple(row) for row in hash_state.multipliers.cpu().tolist()),
        num_embeddings=layout.num_embeddings,
        compressed_vocab_size=layout.compressed_vocab_size,
    )
    pad = hash_state.pad_id

    def row_ids(raw: torch.Tensor, layer_id: int, start: int, end: int):
        dead = (raw < 0) | (raw >= token_map.numel()) | image_sentinel_mask(raw)
        compressed = token_map[raw.clamp(0, token_map.numel() - 1)].masked_fill(
            dead, -1
        )
        ids = host.query_row_ids(
            compressed, geometry=geometry, layer_id=layer_id, pad=pad
        )
        return ids[(ids >= start) & (ids < end)]

    def resolve_table(layer: "Engram"):
        return layer.embed_tokens.disk_table

    return tuple(
        PrefetchTarget(
            resolve_table=partial(resolve_table, layer),
            row_ids=partial(
                row_ids,
                layer_id=layout.layer_ids[layer.layer_hash_index],
                start=layer.embed_tokens.vocab_start_idx,
                end=layer.embed_tokens.vocab_end_idx,
            ),
        )
        for layer in layers
    )
