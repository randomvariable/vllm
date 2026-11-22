# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""B12X checkpoint routing with real, small safetensors shards."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import safe_open, save_file

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import (
    get_file_tensor_source,
    is_weight_transfer_active,
)


def test_load_format_selects_native_b12x_without_plugin(monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "")
    options = {"read_mode": "bounce", "io_threads": 2}
    config = LoadConfig(load_format="b12x", model_loader_extra_config=options.copy())
    loader = get_model_loader(config)
    from vllm.model_executor.model_loader.b12x_loader import B12xModelLoader

    assert isinstance(loader, B12xModelLoader)
    assert loader.read_mode == "bounce"
    assert loader.io_threads == 2
    assert config.load_format == "b12x"
    assert config.model_loader_extra_config == options
    assert type(get_model_loader(LoadConfig(load_format="safetensors"))) is (
        DefaultModelLoader
    )
    assert type(get_model_loader(LoadConfig(load_format="mxfp4_csf"))).__name__ == (
        "Mxfp4CsfModelLoader"
    )


@pytest.mark.parametrize(
    ("options", "strategy", "error"),
    [
        ({"read_mode": "mmap"}, None, "read_mode"),
        ({"io_threads": 0}, None, "io_threads"),
        ({"io_threads": True}, None, "io_threads"),
        ({"enable_multithread_load": True}, None, "enable_multithread_load"),
        ({}, "eager", "safetensors_load_strategy"),
    ],
)
def test_unsupported_config_is_rejected(options, strategy, error):
    with pytest.raises(ValueError, match=error):
        get_model_loader(
            LoadConfig(
                load_format="b12x",
                model_loader_extra_config=options,
                safetensors_load_strategy=strategy,
            )
        )


@pytest.fixture(autouse=True)
def _active_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _model_config(path):
    return SimpleNamespace(
        model=str(path),
        revision=None,
        quantization=None,
        enable_cumem_allocator=False,
        is_moe=False,
        multimodal_config=None,
    )


def test_csf_decoder_reads_meta_stream_bytes():
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        csf_weights_transform,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    fixed = torch.full((1, 32), 0x21, dtype=torch.uint8)
    exceptions = torch.empty(0, dtype=torch.uint32)
    streams = [fixed, exceptions]
    descriptors = [torch.empty_like(t, device="meta") for t in streams]

    class Reader:
        def materialize(self, source):
            return streams[0 if source is descriptors[0] else 1]

    transform = csf_weights_transform(["csf"])
    with weight_transfer(Reader()):
        [(name, scale)] = transform(
            zip(
                ("scale.nvfp4_csf_fixed", "scale.nvfp4_csf_exceptions"),
                descriptors,
                strict=True,
            )
        )
    assert name == "scale"
    torch.testing.assert_close(
        scale.view(torch.uint8), torch.tensor([[0x22, 0x23]] * 16, dtype=torch.uint8)
    )


def test_no_pt_fallback(tmp_path):
    torch.save({"weight": torch.arange(2)}, tmp_path / "weights.pt")
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        get_model_loader(LoadConfig(load_format="b12x")).download_model(
            _model_config(tmp_path)
        )


def test_shared_gds_rejects_cumem_before_allocating_weights(tmp_path):
    loader = get_model_loader(
        LoadConfig(load_format="b12x", model_loader_extra_config={"read_mode": "gds"})
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device="cuda:0"), load_config=loader.load_config
    )
    model_config = _model_config(tmp_path)
    model_config.enable_cumem_allocator = True
    with pytest.raises(ValueError, match="does not support the CuMem allocator"):
        loader.load_model(config, model_config)


class _CheckpointModel(torch.nn.Module):
    def __init__(self, root, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, device=device), False)
        self.draft = torch.nn.Embedding(3, 4, device=device)
        self.draft.weight.requires_grad_(False)
        self.secondary_weights = (
            DefaultModelLoader.Source(
                str(root),
                None,
                subfolder="draft",
                prefix="draft.",
                is_unused_weight=lambda name: name != "weight",
            ),
        )
        self.checkpoint_file_weight_filter = lambda name: name == "table"
        self.seen = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            self.seen.append(name)
            if name == "table":
                self.table = get_file_tensor_source(weight)
                assert weight.is_meta and self.table is not None
            else:
                assert weight.is_meta, "ordinary weights must not be staged"
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded


@pytest.fixture
def indexed_checkpoint(tmp_path):
    primary = tmp_path / "primary.safetensors"
    save_file({"weight": torch.arange(4.0), "table": torch.arange(8)}, primary)
    save_file({"weight": torch.full((4,), -1.0)}, tmp_path / "wrong.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": primary.name, "table": primary.name}})
    )
    draft = tmp_path / "draft"
    draft.mkdir()
    save_file({"weight": torch.arange(12.0).view(3, 4)}, draft / "draft.safetensors")
    (draft / "unused.safetensors").write_bytes(b"must not be opened")
    (draft / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "draft.safetensors",
                    "unused.weight": "unused.safetensors",
                }
            }
        )
    )
    return tmp_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA O_DIRECT")
@pytest.mark.parametrize("read_mode", ["auto", "bounce"])
def test_indexed_sources_and_disk_table_survive_reader_close(
    indexed_checkpoint, read_mode
):
    root = indexed_checkpoint
    loader = get_model_loader(
        LoadConfig(
            load_format="b12x",
            device="cuda:0",
            model_loader_extra_config={"read_mode": read_mode},
        )
    )
    model = _CheckpointModel(root, "cuda:0")
    loader.load_weights(model, _model_config(root))
    assert model.seen == ["table", "weight", "draft.weight"]
    torch.testing.assert_close(model.weight.cpu(), torch.arange(4.0))
    torch.testing.assert_close(model.draft.weight.cpu(), torch.arange(12.0).view(3, 4))
    assert model.table is not None
    assert Path(model.table.path) == root / "primary.safetensors"
    assert model.table.offset > 8
    assert model.table.shape == (8,)
    assert model.table.dtype == torch.int64
    with open(model.table.path, "rb") as file:
        file.seek(model.table.offset)
        data = torch.frombuffer(bytearray(file.read(64)), dtype=torch.int64)
    torch.testing.assert_close(data, torch.arange(8))
    assert not is_weight_transfer_active()


@pytest.mark.parametrize("fail_loading", [False, True])
def test_finishes_once_and_closes_session_on_error(tmp_path, monkeypatch, fail_loading):
    from vllm.model_executor.model_loader import b12x_loader

    loader = get_model_loader(LoadConfig(load_format="b12x", device="cuda:0"))
    events = []

    class Session:
        def __call__(self, destination, source):
            return False

        def start(self):
            events.append("start")

        def finish(self):
            events.append("finish")

        def stats(self, *, flush):
            assert not flush
            return {"payload_bytes": 0}

    @contextmanager
    def session(*args):
        try:
            yield Session()
        finally:
            events.append("close")

    monkeypatch.setattr(b12x_loader.B12xModelLoader, "_open_session", session)

    class Model(torch.nn.Module):
        def load_weights(self, weights):
            assert is_weight_transfer_active()
            events.append("load")
            if fail_loading:
                raise RuntimeError("checkpoint read failed")
            return set()

    if fail_loading:
        with pytest.raises(RuntimeError, match="checkpoint read failed"):
            loader.load_weights(Model(), _model_config(tmp_path))
        assert events == ["load", "close"]
    else:
        loader.load_weights(Model(), _model_config(tmp_path))
        assert events == ["load", "finish", "close"]
    assert not is_weight_transfer_active()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA O_DIRECT")
def test_gdn_convolution_shards_read_into_final_parameter_slices(tmp_path):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )

    path = tmp_path / "conv.safetensors"
    checkpoint = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    save_file({"conv.weight": checkpoint}, path)
    route = mamba_v2_sharded_weight_loader([(8, 0, 0), (4, 2, 1)], tp_size=2, tp_rank=1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.full((6, 2), -1.0, device="cuda"), False
            )

        def load_weights(self, weights):
            for name, weight in weights:
                assert name == "conv.weight"
                route(self.weight.data, weight)
            return {"weight"}

    loader = get_model_loader(
        LoadConfig(
            load_format="b12x",
            device="cuda:0",
            model_loader_extra_config={"read_mode": "bounce"},
        )
    )
    model = Model()
    loader.load_weights(model, _model_config(tmp_path))
    with safe_open(path, framework="pt") as reader:
        torch.testing.assert_close(
            model.weight.cpu(), reader.get_tensor("conv.weight")[4:]
        )
