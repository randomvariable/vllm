# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Extracted verbatim from ATOM atom/plugin/vllm/gdn_backend.py.
# Source: https://github.com/ROCm/ATOM/blob/main/atom/plugin/vllm/gdn_backend.py
# Lines 31-127 (AtomGDNAttentionMetadataBuilder class, _compact_full_graph_decode_metadata method).
#
# This file contains ONLY the GDN full-graph decode metadata compaction logic.
# The registration and backend class are NOT included — only the compaction
# method that strips padded cudagraph rows from decode metadata.

"""GDN full-graph decode metadata compaction.

When vLLM runs GDN attention under CUDA graphs, the decode batch is padded to
a fixed size. Some of the padded rows may have query_len == 0 (no real decode
request), but they still occupy state slots. This compaction method:

  1. Identifies which decode rows are "real" (query_len > 0) vs padded.
  2. Compacts the query_start_loc, state_indices, and num_decodes to cover
     only the real decode rows.
  3. Fills the remaining slots with PAD_SLOT_ID so the cudagraph kernel sees
     valid-but-unused entries.

This is critical for correctness when the cudagraph batch size exceeds the
actual number of decode requests — without compaction, padded rows would
consume state slots and corrupt the SSM state indexing.

gfx1151 applicability:
  This compaction runs on CPU (query_start_loc_cpu) and GPU (state_indices).
  It is portable to any AMD GPU that supports GDN attention with full CUDA
  graphs. The logic is architecture-agnostic.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor


def _compact_full_graph_decode_metadata(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    attn_metadata: GDNAttentionMetadata,
) -> None:
    """Compact padded cudagraph decode rows from GDN metadata.

    Called after super().build() in AtomGDNAttentionMetadataBuilder.build().
    Only runs when use_full_cuda_graph is True and there are decode requests.

    Args:
        common_attn_metadata: vLLM's common metadata (query_start_loc, block_table, etc.)
        attn_metadata: GDN-specific metadata to compact in-place.
    """
    if not getattr(self, "use_full_cuda_graph", False):
        return
    if (
        attn_metadata.num_prefills != 0
        or attn_metadata.num_spec_decodes != 0
        or attn_metadata.num_decodes <= 0
    ):
        return

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    if query_start_loc_cpu is None or query_start_loc_cpu.numel() <= 1:
        return

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    real_decode_mask_cpu = query_lens_cpu > 0
    real_num_decodes = int(real_decode_mask_cpu.sum().item())
    if real_num_decodes == attn_metadata.num_decodes:
        return

    batch_size = int(common_attn_metadata.num_actual_tokens)
    if batch_size > self.decode_cudagraph_max_bs:
        return

    query_start_loc = common_attn_metadata.query_start_loc
    block_table_tensor = mamba_get_block_table_tensor(
        common_attn_metadata.block_table_tensor,
        common_attn_metadata.seq_lens,
        self.kv_cache_spec,
        self.vllm_config.cache_config.mamba_cache_mode,
    )

    state_indices = self.non_spec_state_indices_tensor[:batch_size]
    if real_num_decodes > 0:
        real_decode_mask = real_decode_mask_cpu.to(
            query_start_loc.device, non_blocking=True
        )
        state_indices[:real_num_decodes].copy_(
            block_table_tensor[real_decode_mask, 0], non_blocking=True
        )
    state_indices[real_num_decodes:].fill_(PAD_SLOT_ID)

    compact_query_start_loc_cpu = torch.zeros(
        real_num_decodes + 1, dtype=torch.int32
    )
    if real_num_decodes > 0:
        torch.cumsum(
            query_lens_cpu[real_decode_mask_cpu].to(torch.int32),
            dim=0,
            out=compact_query_start_loc_cpu[1:],
        )

    query_start_loc_buf = self.non_spec_query_start_loc[: batch_size + 1]
    query_start_loc_buf[: real_num_decodes + 1].copy_(
        compact_query_start_loc_cpu.to(query_start_loc.device, non_blocking=True),
        non_blocking=True,
    )
    terminal = query_start_loc_buf[real_num_decodes]
    query_start_loc_buf[real_num_decodes + 1 :].fill_(terminal)

    attn_metadata.num_decodes = real_num_decodes
    attn_metadata.num_decode_tokens = int(
        query_lens_cpu[real_decode_mask_cpu].sum().item()
    )
    attn_metadata.non_spec_state_indices_tensor = state_indices
    attn_metadata.non_spec_query_start_loc = query_start_loc_buf
