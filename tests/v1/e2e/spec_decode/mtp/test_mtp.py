# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import pytest
import torch

from tests.utils import large_gpu_mark, single_gpu_only
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False, 0.5),  # ref: 65%-70%
        pytest.param(
            ("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["mimo", "deepseek", "qwen3_5-hybrid", "gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int] | tuple[str, str, int, str],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
):
    """
    Compare the outputs of a original LLM and a speculative LLM
    which should be the same when using MTP speculative decoding. Due to some variance
    in the engine, it is possible for some outputs to differ, so we expect that at least
    6/10 output tokens match exactly, and that the GSM8k accuracy is above a precomputed
    reference threshold for each model.
    """
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if len(model_setup) == 4:
            method, model_name, tp_size, draft_model = model_setup
        else:
            method, model_name, tp_size = model_setup
            draft_model = None
        _skip_if_insufficient_gpus_for_tp(tp_size)

        if "Qwen3.5" in model_name and os.environ.get("VLLM_USE_V2_MODEL_RUNNER"):
            pytest.skip(
                "Model Runner V2 does not yet support hybrid models "
                "(Qwen3.5 mixes Mamba-style GDN with attention layers)."
            )

        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"

        # Skip multimodal profiling for models that don't need it in this test.
        extra_kwargs: dict[str, Any] = {}
        if "Qwen3.5" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        elif "gemma-4" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

        if draft_model is not None and "gemma-4" in draft_model:
            import transformers
            from packaging.version import Version

            if Version(transformers.__version__) < Version("5.8.0"):
                pytest.skip(
                    "Gemma4 MTP assistant requires transformers>=5.8.0, "
                    f"got {transformers.__version__}"
                )

        ref_llm = LLM(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        evaluate_llm_for_gsm8k(
            ref_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        del ref_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

        speculative_config: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": 1,
            "max_model_len": 2048,
        }
        if draft_model is not None:
            speculative_config["model"] = draft_model
            speculative_config["num_speculative_tokens"] = 2

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config=speculative_config,
            max_model_len=2048,
            attention_backend=attn_backend,
            **extra_kwargs,
        )
        # MTP supports async scheduling; assert it is active by default.
        assert spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        evaluate_llm_for_gsm8k(
            spec_llm, expected_accuracy_threshold=expected_accuracy_threshold
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.8 * len(ref_outputs))
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@single_gpu_only
@large_gpu_mark(min_gb=20)
def test_mtp_hybrid_prefix_cache_reuse():
    """MTP on Qwen3.5 reuses aligned hybrid-model prefix-cache blocks."""
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1":
        pytest.skip("Qwen3.5 hybrid models are unsupported by Model Runner V2")

    model = "Qwen/Qwen3.5-0.8B-Base"
    engine_kwargs = {
        "model": model,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "trust_remote_code": True,
        "mamba_cache_mode": "align",
        "speculative_config": {
            "method": "mtp",
            "model": model,
            "num_speculative_tokens": 1,
        },
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=2)
    cold_llm = LLM(**engine_kwargs, enable_prefix_caching=False)
    try:
        tokenizer = cold_llm.get_tokenizer()
        block_size = cold_llm.llm_engine.vllm_config.cache_config.block_size
        prefix_ids = tokenizer.encode(
            "Prefix cache regression. " * (block_size * 2)
        )[: block_size * 2]
        suffix_a = tokenizer.encode(" Answer with alpha.")
        suffix_b = tokenizer.encode(" Answer with beta.")
        prompt_a = TokensPrompt(prompt_token_ids=prefix_ids + suffix_a)
        prompt_b = TokensPrompt(prompt_token_ids=prefix_ids + suffix_b)
        cold_b = cold_llm.generate([prompt_b], sampling_params)[0]
    finally:
        del cold_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()

    llm = LLM(**engine_kwargs, enable_prefix_caching=True)
    try:
        llm.generate([prompt_a], sampling_params)
        first_b = llm.generate([prompt_b], sampling_params)[0]
        second_b = llm.generate([prompt_b], sampling_params)[0]
        third_b = llm.generate([prompt_b], sampling_params)[0]

        assert first_b.num_cached_tokens >= len(prefix_ids)
        assert second_b.num_cached_tokens == first_b.num_cached_tokens
        assert third_b.num_cached_tokens == second_b.num_cached_tokens
        cold_tokens = cold_b.outputs[0].token_ids
        assert first_b.outputs[0].token_ids == cold_tokens
        assert second_b.outputs[0].token_ids == cold_tokens
        assert third_b.outputs[0].token_ids == cold_tokens
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
