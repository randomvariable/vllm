# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.warmup.kernel_warmup as kernel_warmup_module
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _qwen_gdn_warmup_config,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
    _warm_layer_norm_kernel,
)
from vllm.platforms import current_platform

_GATED_RUNNER_WARMUPS = frozenset(
    {
        "qwen_triton_warmup",
        "qwen_vl_triton_warmup",
        "mamba_triton_warmup",
        "kimi_k3_triton_warmup",
        "watermark_sample_warmup",
        "qwen4_exp_qsa_triton_warmup",
    }
)


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
        norm_weight=torch.ones(v, dtype=torch.bfloat16, device=device),
        norm_bias=None,
        norm_eps=1e-6,
        norm_group_size=v,
    )


class _PerHeadNorm:
    """Mirrors RMSNormGated(head_v_dim, group_size=None) in qwen_gdn_linear_attn."""

    def __init__(self, v: int) -> None:
        self.weight = torch.ones(v, dtype=torch.bfloat16)
        self.bias = None
        self.eps = 1e-6
        self.group_size = None
        self.norm_before_gate = True
        self.activation = "silu"


def _fake_gdn_layer(h: int, hv: int, k: int, v: int, tp_size: int) -> SimpleNamespace:
    conv_dim = 2 * (h // tp_size) * k + (hv // tp_size) * v
    return SimpleNamespace(
        num_k_heads=h,
        num_v_heads=hv,
        head_k_dim=k,
        head_v_dim=v,
        conv_kernel_size=4,
        tp_size=tp_size,
        norm=_PerHeadNorm(v),
        A_log=torch.zeros(hv // tp_size, dtype=torch.float32),
        dt_bias=torch.zeros(hv // tp_size, dtype=torch.float32),
        kv_cache=(
            torch.zeros(2, conv_dim, 3, dtype=torch.bfloat16),
            torch.zeros(2, hv // tp_size, k, v, dtype=torch.float32),
        ),
    )


def test_qwen_gdn_norm_warmup_matches_per_head_weight_width() -> None:
    # Qwen GDN normalizes per value head, so the warmed activation must be as wide
    # as the norm weight. Sizing it from the stacked heads makes layer_norm_fwd
    # reject the weight and aborts EngineCore init.
    hv, v = 48, 128
    config = _qwen_gdn_warmup_config(
        {"layers.0.linear_attn": _fake_gdn_layer(32, hv, 128, v, tp_size=1)}
    )
    assert config is not None
    assert config.norm_group_size == config.norm_weight.shape[0] == v


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    _warm_layer_norm_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.parametrize("enable_jit_warmup", [False, True])
def test_kernel_warmup_honours_jit_warmup_flag(monkeypatch, enable_jit_warmup):
    """Runner-owned Triton warmups launch kernels, so they must obey the same
    opt-out as the registry warmups instead of running unconditionally."""
    ran: list[str] = []

    def record(name):
        def _warmup(*args, **kwargs):
            ran.append(name)

        return _warmup

    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(
            jit_warmup_registry=SimpleNamespace(warmup=record("registry"))
        ),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(
                enable_jit_warmup=enable_jit_warmup,
                enable_cutedsl_warmup=False,
            ),
            model_config=SimpleNamespace(),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        get_model=lambda: None,
    )

    for name in _GATED_RUNNER_WARMUPS:
        monkeypatch.setattr(kernel_warmup_module, name, record(name))
    for name in ("_warmup_bf16x3_router_gemm", "_warmup_ll_bf16_router_gemm"):
        monkeypatch.setattr(kernel_warmup_module, name, record(name))

    kernel_warmup_module.kernel_warmup(worker, process_local_only=True)  # type: ignore[arg-type]

    assert set(ran) & _GATED_RUNNER_WARMUPS == (
        set(_GATED_RUNNER_WARMUPS) if enable_jit_warmup else set()
    )
