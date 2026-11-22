# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision controls for an untied token embedding and vocabulary projection."""

import pytest
import torch

from vllm.compilation.backends import set_model_tag
from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

VOCAB, HIDDEN = 251, 256


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_head_flag_does_not_quantize_embedding_or_owned_draft(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("VLLM_MXFP8_LM_HEAD", raising=False)
    else:
        monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", flag)
    monkeypatch.setenv("VLLM_LM_HEAD_A16", "1")
    embed = VocabParallelEmbedding(VOCAB, HIDDEN, torch.bfloat16, disable_tp=True)
    with set_model_tag("dspark_head"):
        draft = ParallelLMHead(
            VOCAB, HIDDEN, params_dtype=torch.bfloat16, disable_tp=True
        )
    for layer in (embed, draft):
        assert isinstance(layer.quant_method, UnquantizedEmbeddingMethod)
        assert layer.weight.dtype == torch.bfloat16
    if flag != "1":
        head = ParallelLMHead(
            VOCAB, HIDDEN, params_dtype=torch.bfloat16, disable_tp=True
        )
        assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)
        assert head.weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("flag", "mode"), [(None, "a16"), ("1", "a16"), ("0", "quantized")]
)
@torch.inference_mode()
def test_target_head_logits(default_vllm_config, dist_init, monkeypatch, flag, mode):
    supported, reason = B12xMxfp8LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    from types import SimpleNamespace

    monkeypatch.setattr(
        default_vllm_config,
        "model_config",
        SimpleNamespace(dtype=torch.bfloat16, head_dtype=torch.bfloat16),
    )
    monkeypatch.setenv("VLLM_B12X_DENSE_ACTIVATION_MODE", "a16")
    monkeypatch.setenv("VLLM_B12X_MXFP8_ACTIVATION_MODE", "quantized")
    if flag is None:
        monkeypatch.delenv("VLLM_LM_HEAD_A16", raising=False)
    else:
        monkeypatch.setenv("VLLM_LM_HEAD_A16", flag)
    generator = torch.Generator().manual_seed(13)
    weights = (torch.randn(VOCAB, HIDDEN, generator=generator) * 0.02).to(
        "cuda", torch.bfloat16
    )
    tokens = torch.arange(17, device="cuda")
    kwargs = dict(params_dtype=torch.bfloat16, disable_tp=True, prefix="lm_head")
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "0")
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, torch.bfloat16, disable_tp=True)
        bf16_head = ParallelLMHead(VOCAB, HIDDEN, **kwargs)
    embed.weight_loader(embed.weight, weights)
    bf16_head.weight_loader(bf16_head.weight, weights.roll(1, dims=0))
    hidden = embed(tokens)
    logits = LogitsProcessor(VOCAB)
    expected = logits(bf16_head, hidden)

    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "1")
    with torch.device("cuda"):
        head = ParallelLMHead(VOCAB, HIDDEN, **kwargs)
    assert isinstance(head.quant_method, Mxfp8OnlineLinearMethod)
    assert isinstance(head.quant_method.kernel, B12xMxfp8LinearKernel)
    with torch.device("cuda"):
        head.weight.weight_loader(head.weight, weights.roll(1, dims=0))
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(head, default_vllm_config.model_config)
    packed = head.b12x_mxfp8_packed_weight.weight
    assert packed.values.dtype == torch.float8_e4m3fn
    assert head.b12x_activation_mode == mode
    actual = logits(head, hidden)
    assert actual.shape == expected.shape == (tokens.numel(), VOCAB)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    error_rms = (actual.float() - expected.float()).square().mean().sqrt().item()
    reference_rms = expected.float().square().mean().sqrt().item()
    print(
        f"{mode}: logit RMS error={error_rms:.7f}, "
        f"relative={error_rms / reference_rms:.4%}"
    )
    # E4M3 block rounding affects weights (and activations in W8A8).
    # Compare the full distribution rather than dividing by individual near-zero logits.
    assert error_rms <= max(1e-5, 0.05 * reference_rms)
