# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.config.engram import EngramConfig
from vllm.models.deepseek_v41.common import engram


def test_disk_offload_rejects_microbatching():
    parallel = SimpleNamespace(
        dp_shared_memory=False,
        data_parallel_size=1,
        enable_elastic_ep=False,
        use_ubatching=True,
    )
    with pytest.raises(ValueError, match="microbatching"):
        EngramConfig(disk_offload=True).verify_parallel_config(parallel)


class ReferenceKernel:
    def __getitem__(self, grid):
        def run(
            weight, scales, ids, dst, out, start, end, rows, stride_t, stride_h, **kw
        ):
            heads = kw["HEAD_START"] + torch.arange(kw["LOCAL_HEADS"])
            valid_heads = heads < kw["TOTAL_HEADS"]
            selected = torch.full((ids.shape[0], len(heads)), -1, dtype=torch.long)
            selected[:, valid_heads] = ids[:, heads[valid_heads]].long()
            valid = (selected >= start) & (selected < end)
            local = (selected - start).masked_fill(~valid, 0)
            values = weight.float()[local]
            scale = (scales[local].int() << 23).view(torch.float32)
            result = values.unflatten(-1, (-1, kw["QUANT_BLOCK"])) * scale.unsqueeze(-1)
            out.copy_(
                result.flatten(-2)
                .to(torch.bfloat16)
                .masked_fill(~valid.unsqueeze(-1), 0)
            )

        return run


@pytest.mark.parametrize("rank", [0, 1])
def test_disk_lookup_matches_resident_checkpoint_tp_shard(tmp_path, monkeypatch, rank):
    monkeypatch.setattr(engram, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=8),
    )
    monkeypatch.setattr(
        engram,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            load_config=SimpleNamespace(load_format="safetensors"),
        ),
    )
    monkeypatch.setattr(engram, "_engram_lookup_kernel", ReferenceKernel())
    weights = (torch.arange(15 * 32).reshape(15, 32).float() % 8).to(
        torch.float8_e4m3fn
    )
    scales = torch.full((15, 1), 127, dtype=torch.uint8)
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "layers.1.engram.embed.weight": weights,
            "layers.1.engram.embed.scale": scales.view(torch.float8_e8m0fnu),
        },
        path,
    )
    resident = engram.ParallelEngramEmbedding(15, 32, (3, 3, 3, 3, 3))
    disk = engram.ParallelEngramEmbedding(15, 32, (3, 3, 3, 3, 3), disk_offload=True)
    with safe_open(path, framework="pt") as checkpoint:
        for table in (resident, disk):
            table.weight.weight_loader(
                table.weight, checkpoint.get_tensor("layers.1.engram.embed.weight")
            )
            table.weight_scale_inv.weight_loader(
                table.weight_scale_inv,
                checkpoint.get_tensor("layers.1.engram.embed.scale"),
            )
            table.process_weights_after_loading()
    assert disk.weight.numel() == disk.weight_scale_inv.numel() == 0
    ids = torch.tensor(
        [[0, 3, 6, 9, 12], [2, 4, 7, 10, 14], [-1, 3, 6, 9, 12]], dtype=torch.int32
    )
    expected = torch.empty((3, 3, 32), dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    resident.lookup(ids, expected)
    disk._disk.start(ids)
    compact = disk._disk.finish()
    disk.lookup(compact, actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    disk._disk.start(ids.flip(0))
    compact = disk._disk.finish()
    disk.lookup(compact, actual)
    resident.lookup(ids.flip(0), expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
