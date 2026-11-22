# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from pydantic import TypeAdapter

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.engram import EngramConfig
from vllm.distributed import parallel_state
from vllm.model_executor.layers import linear
from vllm.models.deepseek_v41.common.engram import Engram, EngramLayout


def _make_engram(settings, sequence_parallel=False, cls=Engram):
    config = SimpleNamespace(
        hidden_size=2,
        hc_mult=1,
        rms_norm_eps=1e-6,
        engram_layer_ids=[0],
        engram_num_embeddings=[16],
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_head_dim=2,
        engram_compressed_vocab_size=16,
        engram_vocab_size=4,
        engram_pad_token_id=0,
    )
    vllm_config = VllmConfig()
    vllm_config.engram_config = settings
    with set_current_vllm_config(vllm_config):
        return cls(
            config,
            quant_config=None,
            layout=EngramLayout(config),
            layer_hash_index=0,
            use_sequence_parallel=sequence_parallel,
            prefix="layers.0.engram",
        )


@pytest.fixture
def projection_inputs(monkeypatch):
    # Stage precomputed rank-local embedding rows instead of running the GPU
    # lookup. Leave Engram's projection construction and forward intact.
    x = torch.tensor([[1.0, 2.0, -1.0, 3.0], [-2.0, 1.0, 0.0, 4.0]])
    weight = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)

    def create_embedding(self, layout, layer_hash_index):
        return SimpleNamespace(tp_size=2, dp_size=1, n_hash_cols=layout.n_hash_cols)

    monkeypatch.setattr(Engram, "_create_embedding", create_embedding)
    monkeypatch.setattr(Engram, "_init_staging", lambda self, *_: None)
    monkeypatch.setattr(Engram, "_post_wkv", lambda self, hidden, kv, *_: kv)
    return x, weight


@pytest.mark.parametrize(
    "config_json,projection_tp",
    [
        ("{}", False),
        ('{"projection_tp":false}', False),
        ('{"projection_tp":true}', True),
    ],
)
def test_projection_config(config_json, projection_tp):
    settings = TypeAdapter(EngramConfig).validate_json(config_json)
    assert settings.projection_tp is projection_tp


@pytest.mark.parametrize(
    "tp_size,sequence_parallel", [(2, False), (2, True), (1, False)]
)
@pytest.mark.parametrize("projection_tp", [False, True])
def test_projection_load_and_forward(
    monkeypatch, projection_inputs, tp_size, sequence_parallel, projection_tp
):
    x, weight = projection_inputs
    settings = EngramConfig(projection_tp=projection_tp)
    sharded = projection_tp and tp_size > 1 and not sequence_parallel
    group = SimpleNamespace(world_size=tp_size, rank_in_group=0)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)

    for rank in range(tp_size):
        group.rank_in_group = rank
        engram = _make_engram(settings, sequence_parallel)
        projection = engram.wkv
        local_weight = weight.chunk(tp_size)[rank] if sharded else weight
        assert projection.weight.shape == local_weight.shape
        projection.weight.weight_loader(projection.weight, weight)
        torch.testing.assert_close(projection.weight, local_weight, atol=0, rtol=0)
        projection.quant_method.process_weights_after_loading(projection)

        # With SP each rank sees different tokens. Such outputs cannot be
        # gathered across the output axis.
        tokens = x[rank : rank + 1] if sequence_parallel else x
        expected = tokens @ weight.T
        gathered: list[bool] = []

        def gather(
            local,
            tokens=tokens,
            local_weight=local_weight,
            gathered=gathered,
            rank=rank,
        ):
            assert sharded, "replicated WKV must not gather"
            torch.testing.assert_close(local, tokens @ local_weight.T, atol=0, rtol=0)
            gathered.append(True)
            parts = [tokens @ part.T for part in weight.chunk(tp_size)]
            parts[rank] = local
            return torch.cat(parts, dim=-1)

        monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather)
        torch.testing.assert_close(projection(tokens), expected, atol=0, rtol=0)
        assert len(gathered) == int(sharded)

        if tp_size == 2 and not sequence_parallel:
            gathered.clear()

            rows = x.chunk(tp_size, dim=-1)[rank].contiguous()

            def gather_columns(local, rows=rows, gather=gather):
                if local.data_ptr() == rows.data_ptr():
                    return x
                return gather(local)

            monkeypatch.setattr(
                engram, "_ready_rows", lambda _, rows=rows: rows.unflatten(1, (1, 2))
            )
            engram.set_column_gather(gather_columns)
            torch.testing.assert_close(
                engram(torch.empty(2, 1, 2), torch.zeros(2, 2, dtype=torch.long)),
                expected,
                atol=0,
                rtol=0,
            )
            assert len(gathered) == int(sharded)


def test_specialized_projection_cannot_silently_ignore_replication(
    monkeypatch, projection_inputs
):
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)

    class ShardedEngram(Engram):
        def _create_wkv(self, input_size, quant_config, prefix):
            # The ROCm specialized projection is built sharded independently
            # of the common constructor.
            return linear.ColumnParallelLinear(input_size, 4, bias=False)

    with pytest.raises(ValueError, match="projection_tp=False.*not supported"):
        _make_engram(EngramConfig(), cls=ShardedEngram)
    assert (
        _make_engram(EngramConfig(projection_tp=True), cls=ShardedEngram).wkv.tp_size
        == 2
    )
