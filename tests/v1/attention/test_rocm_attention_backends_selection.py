# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend selectors."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

# ROCm-specific attention backend selection tests
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.model_config.dtype = torch.float16
    config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    config.cache_config.block_size = 16
    return config


@pytest.fixture
def mock_get_cdna_version():
    """Mock cdna version arch detection to return True."""
    with patch("vllm.platforms.rocm.get_cdna_version", return_value=3):
        yield


@pytest.mark.parametrize(
    "env_vars, selected_backend, expected_backend_path",
    [
        # Test Case: Explicit FLEX_ATTENTION backend
        (
            {},
            "FLEX_ATTENTION",
            AttentionBackendEnum.FLEX_ATTENTION.get_path(),
        ),
        # Test Case 1: Default (no env vars, no explicit backend)
        (
            {},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 2: Explicit TRITON_ATTN backend
        (
            {},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 3: Explicit ROCM_ATTN backend
        (
            {},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 4: Explicit ROCM_AITER_FA backend
        (
            {},
            "ROCM_AITER_FA",
            AttentionBackendEnum.ROCM_AITER_FA.get_path(),
        ),
        # Test Case 5: Explicit ROCM_AITER_UNIFIED_ATTN backend
        (
            {},
            "ROCM_AITER_UNIFIED_ATTN",
            AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path(),
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 8: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MHA=0
        (
            {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "0"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 9: VLLM_ROCM_USE_AITER=1 + explicit ROCM_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
    ],
)
def test_standard_attention_backend_selection(
    env_vars,
    selected_backend,
    expected_backend_path,
    mock_vllm_config,
    mock_get_cdna_version,
    monkeypatch,
):
    """Test standard attention backend selection with various configurations."""
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars to ensure they're picked up
    # Reload envs to pick up new environment variables
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Convert string backend to enum if provided
    backend_enum = None
    if selected_backend:
        backend_enum = getattr(AttentionBackendEnum, selected_backend)

    # Get the backend class path
    from vllm.platforms.rocm import RocmPlatform

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=backend_enum, attn_selector_config=attn_selector_config
    )

    assert backend_path == expected_backend_path


@pytest.mark.parametrize(
    "env_vars, selected_backend, block_size, expected_backend_path, should_raise",
    [
        # Test Case 1: TRITON_MLA with block_size != 1
        (
            {},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 2: TRITON_MLA with block_size == 1 (should raise)
        (
            {},
            "TRITON_MLA",
            1,
            None,
            True,
        ),
        # Test Case 3: ROCM_AITER_MLA with block_size == 1
        (
            {},
            "ROCM_AITER_MLA",
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 4: ROCM_AITER_MLA with block_size != 1 (should raise)
        (
            {},
            "ROCM_AITER_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 5: VLLM_ROCM_USE_AITER=1 with block_size == 1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1 with block_size == 16
        # (should use ROCM_AITER_MLA now, as it supports block_size 16)
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_MLA
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 8: Explicit ROCM_AITER_TRITON_MLA
        (
            {},
            "ROCM_AITER_TRITON_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_TRITON_MLA.get_path(),
            False,
        ),
    ],
)
def test_mla_backend_selection(
    env_vars,
    selected_backend,
    block_size,
    expected_backend_path,
    should_raise,
    mock_vllm_config,
    mock_on_mi3xx,
    monkeypatch,
):
    """Test MLA backend selection with various configurations.

    These cases cover env-var and block-size semantics, not architecture, so
    they assume a mi3xx host: the AITER MLA backends gate on it via
    supports_compute_capability. Architecture behaviour is covered separately
    by test_aiter_mla_skipped_on_unsupported_arch and friends.
    """
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars
    # Reload envs
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Mock is_aiter_mla_enabled based on env vars and block_size
    aiter_enabled = env_vars.get("VLLM_ROCM_USE_AITER") == "1"

    mock_rocm_ops = MagicMock()
    mock_rocm_ops.is_mla_enabled.return_value = aiter_enabled
    mock_aiter_module = MagicMock()
    mock_aiter_module.rocm_aiter_ops = mock_rocm_ops

    with patch.dict("sys.modules", {"vllm._aiter_ops": mock_aiter_module}):
        # Convert string backend to enum if provided
        backend_enum = None
        if selected_backend:
            backend_enum = getattr(AttentionBackendEnum, selected_backend)

        from vllm.platforms.rocm import RocmPlatform

        if should_raise:
            with pytest.raises(ValueError):
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                )
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                )
                backend_path = RocmPlatform.get_attn_backend_cls(
                    selected_backend=backend_enum,
                    attn_selector_config=attn_selector_config,
                )

        else:
            attn_selector_config = AttentionSelectorConfig(
                head_size=128,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )

            backend_path = RocmPlatform.get_attn_backend_cls(
                selected_backend=backend_enum, attn_selector_config=attn_selector_config
            )

            assert backend_path == expected_backend_path


def test_aiter_fa_requires_mi3xx(mock_vllm_config):
    """Test that ROCM_AITER_FA requires CDNA3+ architecture."""
    from vllm.platforms.rocm import RocmPlatform

    # Mock cdna version to return 1 (used by supports_compute_capability)
    with (
        patch("vllm.platforms.rocm.get_cdna_version", return_value=1),
        pytest.raises(
            ValueError,
            match="compute capability not supported",
        ),
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_AITER_FA,
            attn_selector_config=attn_selector_config,
        )


def _mla_selector_config(block_size: int = 16) -> AttentionSelectorConfig:
    return AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=block_size,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
    )


def test_aiter_mla_skipped_on_unsupported_arch(mock_vllm_config, monkeypatch):
    """Automatic MLA selection must fall through to TRITON_MLA off mi3xx.

    AITER's MLA kernels only exist for gfx942/gfx950, but AITER as a library is
    also admitted on gfx1151 for its non-MLA kernels. With the AITER MLA env
    gate on, automatic selection must therefore skip ROCM_AITER_MLA on an
    unsupported arch instead of picking a backend whose kernels are absent.
    """
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    from vllm.platforms.rocm import RocmPlatform

    mock_rocm_ops = MagicMock()
    mock_rocm_ops.is_mla_enabled.return_value = True

    with (
        patch("vllm._aiter_ops.rocm_aiter_ops", mock_rocm_ops),
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
    ):
        backend_path = RocmPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=_mla_selector_config(),
        )

    assert backend_path != AttentionBackendEnum.ROCM_AITER_MLA.get_path()
    assert backend_path == AttentionBackendEnum.TRITON_MLA.get_path()


def test_aiter_mla_still_selected_on_supported_arch(mock_vllm_config, monkeypatch):
    """No regression on gfx942/gfx950: ROCM_AITER_MLA stays the first choice."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    from vllm.platforms.rocm import RocmPlatform

    mock_rocm_ops = MagicMock()
    mock_rocm_ops.is_mla_enabled.return_value = True

    with (
        patch("vllm._aiter_ops.rocm_aiter_ops", mock_rocm_ops),
        patch("vllm.platforms.rocm.on_mi3xx", return_value=True),
    ):
        backend_path = RocmPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=_mla_selector_config(),
        )

    assert backend_path == AttentionBackendEnum.ROCM_AITER_MLA.get_path()


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.ROCM_AITER_MLA,
        # AiterTritonMLABackend subclasses AiterMLABackend and only overrides
        # the prefill call, still inheriting forward_mqa -> mla_decode_fwd, so
        # it must inherit the arch gate too.
        AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
    ],
)
def test_explicit_aiter_mla_raises_on_unsupported_arch(backend, mock_vllm_config):
    """Explicit selection fails closed rather than silently downgrading."""
    from vllm.platforms.rocm import RocmPlatform

    with (
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
        pytest.raises(ValueError, match="compute capability not supported"),
    ):
        RocmPlatform.get_attn_backend_cls(
            selected_backend=backend,
            attn_selector_config=_mla_selector_config(),
        )


def test_sparse_mla_not_gated_by_dense_arch_check(mock_vllm_config):
    """The dense gate must not leak into the sparse backend.

    ROCM_AITER_MLA_SPARSE has its own (separately tracked) support story; this
    change must leave it exactly as it was.
    """
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    with patch("vllm.platforms.rocm.on_mi3xx", return_value=False):
        assert ROCMAiterMLASparseBackend.supports_compute_capability(
            DeviceCapability(11, 5)
        )


def test_sparse_not_supported(mock_vllm_config):
    """Test that sparse MLA without use_mla flag raises an error."""
    from vllm.platforms.rocm import RocmPlatform

    with pytest.raises(
        ValueError,
        match="No valid attention backend found",
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=True,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=None, attn_selector_config=attn_selector_config
        )


# ---------------------------------------------------------------------------
# Observability of the ROCm custom paged-attention fallback.
#
# ROCM_ATTN reports itself as the selected backend even when its custom paged
# kernel cannot serve the runtime config, in which case it silently runs its
# internal Triton path. unsupported_reason_rocm_custom_paged_attention names the
# specific reason so the fallback is visible in the log.
#
# The reason function must agree with use_rocm_custom_paged_attention: whenever
# the latter says False the former must give a reason, and vice versa. These
# tests pin that agreement as well as the individual messages.
# ---------------------------------------------------------------------------

_IN_ENVELOPE_GFX9 = dict(
    qtype=torch.float16,
    head_size=128,
    block_size=16,
    gqa_ratio=8,
    max_seq_len=4096,
    sliding_window=0,
    kv_cache_dtype="auto",
)


@pytest.fixture
def clear_paged_attn_caches():
    """Clear the ``@cache`` on both envelope helpers around each test.

    Both are keyed on the config args only -- architecture is a module-level
    constant that never changes within a real process, so caching across archs
    cannot happen in production. Tests do patch the arch, so the caches must be
    cleared or a patched-arch call would get a stale answer.
    """
    from vllm.platforms.rocm import (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    )

    for fn in (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    ):
        fn.cache_clear()
    yield
    for fn in (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    ):
        fn.cache_clear()


@pytest.fixture
def as_gfx9():
    """Pretend the module resolved a gfx942-class arch."""
    with (
        patch("vllm.platforms.rocm._ON_GFX9", True),
        patch("vllm.platforms.rocm._ON_GFX1X", False),
        patch("vllm.platforms.rocm._GCN_ARCH", "gfx942"),
    ):
        yield


@pytest.fixture
def as_gfx1x():
    """Pretend the module resolved an RDNA arch (e.g. gfx1151)."""
    with (
        patch("vllm.platforms.rocm._ON_GFX9", False),
        patch("vllm.platforms.rocm._ON_GFX1X", True),
        patch("vllm.platforms.rocm._GCN_ARCH", "gfx1151"),
    ):
        yield


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        # Shared rejections (both architectures).
        ({"sliding_window": 4096}, "sliding window enabled"),
        ({"qtype": torch.float32}, "unsupported query dtype"),
        ({"max_seq_len": 256 * 1024}, "exceeds 128K"),
        # gfx9 envelope: head 64/128, block 16/32, GQA 1-16.
        ({"head_size": 96}, "unsupported head size"),
        ({"block_size": 64}, "unsupported block size"),
        ({"gqa_ratio": 32}, "outside supported range 1-16"),
    ],
)
def test_fallback_reason_gfx9(
    overrides, expected_fragment, as_gfx9, clear_paged_attn_caches
):
    """Each out-of-envelope gfx9 config reports its own specific reason."""
    from vllm.platforms.rocm import unsupported_reason_rocm_custom_paged_attention

    reason = unsupported_reason_rocm_custom_paged_attention(
        **{**_IN_ENVELOPE_GFX9, **overrides}
    )
    assert reason is not None
    assert expected_fragment in reason, reason
    # Must not be the generic message this replaced.
    assert "falling back" not in reason.lower()


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        # RDNA is strictly narrower: head must be 128, block 16, GQA 3-16,
        # no ALiBi, unquantized KV.
        ({"head_size": 64}, "needs 128 on this arch"),
        ({"block_size": 32}, "needs 16 on this arch"),
        ({"gqa_ratio": 1}, "outside supported range 3-16"),
        ({"has_alibi_slopes": True}, "ALiBi slopes not supported"),
        ({"kv_cache_dtype": "fp8"}, "quantized KV cache"),
        ({"has_sinks": True}, "attention sinks enabled"),
    ],
)
def test_fallback_reason_gfx1x(
    overrides, expected_fragment, as_gfx1x, clear_paged_attn_caches
):
    """Each out-of-envelope RDNA config reports its own specific reason."""
    from vllm.platforms.rocm import unsupported_reason_rocm_custom_paged_attention

    reason = unsupported_reason_rocm_custom_paged_attention(
        **{**_IN_ENVELOPE_GFX9, **overrides}
    )
    assert reason is not None
    assert expected_fragment in reason, reason


def test_no_reason_when_in_envelope_gfx9(as_gfx9, clear_paged_attn_caches):
    """In-envelope config yields no reason, i.e. no diagnostic is emitted."""
    from vllm.platforms.rocm import (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    )

    assert unsupported_reason_rocm_custom_paged_attention(**_IN_ENVELOPE_GFX9) is None
    # And the envelope check agrees that the custom kernel is usable.
    assert use_rocm_custom_paged_attention(**_IN_ENVELOPE_GFX9) is True


def test_no_reason_when_in_envelope_gfx1x(as_gfx1x, clear_paged_attn_caches):
    """head=128 / block=16 / GQA 8 is inside the RDNA envelope too."""
    from vllm.platforms.rocm import (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    )

    assert unsupported_reason_rocm_custom_paged_attention(**_IN_ENVELOPE_GFX9) is None
    assert use_rocm_custom_paged_attention(**_IN_ENVELOPE_GFX9) is True


def test_unsupported_architecture_reported(clear_paged_attn_caches):
    """Neither gfx9 nor RDNA: name the architecture rather than a condition."""
    from vllm.platforms.rocm import unsupported_reason_rocm_custom_paged_attention

    with (
        patch("vllm.platforms.rocm._ON_GFX9", False),
        patch("vllm.platforms.rocm._ON_GFX1X", False),
        patch("vllm.platforms.rocm._GCN_ARCH", "gfx1030"),
    ):
        reason = unsupported_reason_rocm_custom_paged_attention(**_IN_ENVELOPE_GFX9)
    assert reason is not None
    assert "unsupported architecture" in reason
    assert "gfx1030" in reason


@pytest.mark.parametrize("arch_fixture", ["as_gfx9", "as_gfx1x"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"sliding_window": 4096},
        {"qtype": torch.float32},
        {"head_size": 96},
        {"block_size": 64},
        {"gqa_ratio": 32},
        {"gqa_ratio": 1},
        {"has_alibi_slopes": True},
        {"kv_cache_dtype": "fp8"},
        {"has_sinks": True},
        {"max_seq_len": 256 * 1024},
    ],
)
def test_reason_agrees_with_envelope_check(
    arch_fixture, overrides, request, clear_paged_attn_caches
):
    """A reason is given exactly when the custom kernel is unusable.

    Guards against the two functions drifting apart: a config the envelope
    rejects with no reason would log "reason unavailable", and a config it
    accepts while a reason exists would log a spurious fallback.
    """
    request.getfixturevalue(arch_fixture)
    from vllm.platforms.rocm import (
        unsupported_reason_rocm_custom_paged_attention,
        use_rocm_custom_paged_attention,
    )

    cfg = {**_IN_ENVELOPE_GFX9, **overrides}
    reason = unsupported_reason_rocm_custom_paged_attention(**cfg)
    usable = use_rocm_custom_paged_attention(
        qtype=cfg["qtype"],
        head_size=cfg["head_size"],
        block_size=cfg["block_size"],
        gqa_ratio=cfg["gqa_ratio"],
        max_seq_len=cfg["max_seq_len"],
        sliding_window=cfg["sliding_window"],
        kv_cache_dtype=cfg["kv_cache_dtype"],
        alibi_slopes=torch.zeros(1) if cfg.get("has_alibi_slopes") else None,
        sinks=torch.zeros(1) if cfg.get("has_sinks") else None,
    )
    assert usable == (reason is None), (
        f"usable={usable} but reason={reason!r} for {overrides}"
    )


def test_diagnostic_logs_once_per_distinct_reason(caplog):
    """The hot-path diagnostic must not spam: once per distinct reason.

    ``logger.info_once`` is backed by an ``lru_cache`` keyed on
    ``(logger, msg, *args)``, so passing the reason as an arg gives one line per
    distinct reason and drops every repeat -- which is what the decode path
    needs, since it reaches this branch on every forward.
    """
    import logging

    from vllm.logger import init_logger

    logger = init_logger("test_rocm_paged_attn_fallback_once")
    msg = "ROCm custom paged attention kernel unavailable (%s); using Triton."

    with caplog.at_level(logging.INFO):
        for _ in range(5):
            logger.info_once(msg, "unsupported head size (96); needs 64 or 128")
        for _ in range(5):
            logger.info_once(msg, "quantized KV cache (fp8) not supported on this arch")

    emitted = [r for r in caplog.records if "ROCm custom paged attention" in r.message]
    assert len(emitted) == 2, [r.message for r in emitted]
    assert all(r.levelno == logging.INFO for r in emitted), (
        "expected INFO: the Triton path is correct for many valid configs and "
        "must not be surfaced as a warning"
    )
