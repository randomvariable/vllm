# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's quantization config asks `nvidia/quantization.py` for
the MXFP8 linear and MXFP4 expert methods on CUDA only; a ROCm GPU, which
can also report capability 9.x, keeps the stock methods."""

import json
from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41 import quant_config
from vllm.models.deepseek_v41.nvidia import quantization


@pytest.mark.parametrize(
    ("checkpoint_method", "expected"),
    [("fp8", "fp8"), ("mxfp4_csf", "mxfp4_csf")],
)
def test_model_config_detects_checkpoint_quantization(
    tmp_path, checkpoint_method, expected
):
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "num_hidden_layers": 6,
        "max_position_embeddings": 4096,
        "quantization_config": {"quant_method": checkpoint_method},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model_config = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True)
    assert model_config.quantization == expected


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(
        quant_config, "current_platform", SimpleNamespace(is_cuda=lambda: False)
    )

    def fail(*args, **kwargs):
        raise AssertionError("the NVIDIA quantization choices ran on ROCm")

    for name in ("mxfp8_linear_method_cls", "mxfp4_moe_method"):
        monkeypatch.setattr(quantization, name, fail)


def test_rocm_keeps_the_stock_linear_method(rocm):
    assert quant_config._mxfp8_linear_method_cls() is ModelOptLinearMethod


def test_rocm_keeps_the_stock_expert_method(rocm, monkeypatch):
    created = {}

    def init(self, moe, activation_key=None):
        created["activation_key"] = activation_key

    monkeypatch.setattr(Mxfp4MoEMethod, "__init__", init)
    method = quant_config._mxfp4_moe_method(SimpleNamespace())
    assert type(method) is Mxfp4MoEMethod
    assert created["activation_key"] == kMxfp8Dynamic


def test_cuda_asks_the_nvidia_module(monkeypatch):
    monkeypatch.setattr(
        quant_config, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(quantization, "mxfp8_linear_method_cls", lambda: "linear")
    monkeypatch.setattr(quantization, "mxfp4_moe_method", lambda moe: ("experts", moe))
    assert quant_config._mxfp8_linear_method_cls() == "linear"
    assert quant_config._mxfp4_moe_method("moe") == ("experts", "moe")
