# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState


def _draft_model_state(max_num_reqs: int = 4) -> Qwen4ExpModelState:
    state = object.__new__(Qwen4ExpModelState)
    state.uses_qsa = True
    state.max_num_reqs = max_num_reqs
    state.qsa_state_slot_ids = torch.arange(max_num_reqs, dtype=torch.int32)
    state._qsa_default_slot_ids = state.qsa_state_slot_ids.clone()
    state.qsa_state_is_fresh = torch.ones(max_num_reqs, dtype=torch.bool)
    state.qsa_num_accepted_tokens = torch.ones(max_num_reqs, dtype=torch.int32)
    state.qsa_committed_num_accepted_tokens_gpu = torch.tensor(
        [5, 6, 7, 8], dtype=torch.int32
    )
    state._qsa_draft_is_prefilling = torch.zeros(max_num_reqs, dtype=torch.bool)
    state._qsa_draft_is_prefilling_gpu = torch.zeros(max_num_reqs, dtype=torch.bool)
    return state


def test_draft_metadata_names_request_slots():
    state = _draft_model_state()

    metadata = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([2, 0], dtype=torch.int64),
        num_reqs=2,
        num_reqs_padded=3,
        draft_index=1,
    )

    assert metadata is not None
    assert metadata.qsa_state_slot_ids is not None
    assert metadata.qsa_state_is_fresh is not None
    assert metadata.qsa_num_accepted_tokens is not None
    assert metadata.qsa_state_slot_ids.tolist() == [2, 0, 2]
    assert metadata.qsa_state_is_fresh.tolist() == [False, False, True]
    assert metadata.qsa_num_accepted_tokens.tolist() == [7, 5, 1]


def test_draft_metadata_keeps_padded_defaults_for_inactive_rows():
    # Dummy batches mark every drafter row -1: none names a request slot.
    state = _draft_model_state()

    metadata = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([-1, -1, -1], dtype=torch.int64),
        num_reqs=3,
        num_reqs_padded=3,
        draft_index=1,
    )

    assert metadata is not None
    assert metadata.qsa_state_slot_ids is not None
    assert metadata.qsa_state_is_fresh is not None
    assert metadata.qsa_num_accepted_tokens is not None
    assert metadata.qsa_state_slot_ids.tolist() == [0, 1, 2]
    assert metadata.qsa_state_is_fresh.tolist() == [True, True, True]
    assert metadata.qsa_num_accepted_tokens.tolist() == [1, 1, 1]
