# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Infer an omitted Qwen4Exp PLE storage dtype from checkpoint headers."""

from pathlib import Path
from typing import Any

from transformers import PreTrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)

# BF16 maps to None: unquantized PLE keeps main's quant-config selection.
_STORAGE_DTYPES = {
    "BF16": None,
    "F8_E4M3": "float8_e4m3fn",
    "U8": "nvfp4",
}


def resolve_ple_embedding_dtype(
    config: PreTrainedConfig,
    config_dict: dict[str, Any],
    model: str | Path,
    revision: str | None,
) -> None:
    """Set `ple_embedding_dtype` from the PLE shard dtype before allocation.

    The b12x PLE path sizes its tables from this field, and checkpoints that
    declare PLE quantization only in `quantization_config` omit it.
    """
    text_config = config.get_text_config()
    text_config_dict = config_dict.get("text_config", config_dict)
    if not getattr(text_config, "ple_layer_ids", None) or text_config_dict.get(
        "ple_embedding_dtype"
    ):
        return

    from vllm.transformers_utils.config import get_safetensors_params_metadata

    source = str(model)
    quant_config = (
        config_dict.get("quantization_config")
        or text_config_dict.get("quantization_config")
        or {}
    )
    if quant_config.get("quant_method") == "mxfp4_csf" and (
        root := quant_config.get("checkpoint_root")
    ):
        # An FP4-CSF serving directory holds metadata only; the checkpoint's
        # tensors, PLE shards included, live under its root.
        source = str(Path(root) / "tensors")
    metadata = get_safetensors_params_metadata(source, revision=revision)
    dtypes = {
        info["dtype"]
        for name, info in metadata.items()
        if ".ple.ple_embedding.ngram_embedding.shard_" in name
        and name.endswith(".weight")
    }
    if not dtypes:
        return
    if len(dtypes) != 1 or not dtypes.issubset(_STORAGE_DTYPES):
        raise ValueError(f"Unsupported PLE checkpoint storage dtypes: {sorted(dtypes)}")
    resolved = _STORAGE_DTYPES[dtypes.pop()]
    if resolved is None:
        return
    text_config.ple_embedding_dtype = resolved
    logger.info("Resolved PLE embedding storage dtype from checkpoint: %s", resolved)
