# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None


def test_probabilistic_draft_sampler_owns_disjoint_philox_offset(monkeypatch):
    captured = {}

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured["positions"] = positions
        captured.update(kwargs)
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(base_spec_module, "gumbel_sample", fake_gumbel_sample)
    positions = torch.tensor([12, 99], dtype=torch.int64)
    active_rows = torch.tensor(2, dtype=torch.int32)
    speculator = object.__new__(_TestSpeculator)
    speculator.use_fp64_gumbel = False

    speculator._sample_probabilistic_draft(
        logits=torch.zeros(2, 5),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
        active_rows=active_rows,
    )

    torch.testing.assert_close(
        captured["positions"], positions + 1 + (1 << 30), rtol=0, atol=0
    )
    assert captured["apply_temperature"] is True
    assert captured["output_processed_logits_active_rows"] is active_rows


def test_ar_probabilistic_draft_uses_shared_sampler(monkeypatch):
    captured = {}

    def fake_sample_probabilistic_draft(
        self,
        logits,
        positions,
        idx_mapping,
        temperature,
        seeds,
        draft_step,
        draft_logits,
        active_rows=None,
    ):
        captured["positions"] = positions
        captured["active_rows"] = active_rows
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(
        DraftModelSpeculator,
        "_sample_probabilistic_draft",
        fake_sample_probabilistic_draft,
    )
    speculator = object.__new__(_TestSpeculator)
    speculator.model = SimpleNamespace(
        compute_logits=lambda hidden_states: torch.zeros(hidden_states.shape[0], 5)
    )
    speculator.active_num_reqs = torch.tensor(2, dtype=torch.int32)
    speculator.use_fp64_gumbel = False
    positions = torch.tensor([12, 99], dtype=torch.int64)

    speculator.sample_draft(
        hidden_states=torch.zeros(2, 3),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
    )

    assert captured["positions"] is positions
    assert captured["active_rows"] is speculator.active_num_reqs
