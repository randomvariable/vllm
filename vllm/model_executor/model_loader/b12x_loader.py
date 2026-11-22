# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import dataclasses
import heapq
import json
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import regex as re
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.weight_utils import (
    file_source_tensor,
    safetensors_file_sources,
)
from vllm.model_executor.weight_transfer import finish_weight_transfers, weight_transfer

logger = init_logger(__name__)


@contextmanager
def _ipc_weight_allocations():
    original = torch._C._accelerator_getAllocatorSettings()
    configured = re.sub(r"(\bexpandable_segments\s*:\s*)True\b", r"\1False", original)
    if configured != original:
        torch._C._accelerator_setAllocatorSettings(configured)
    try:
        yield
    finally:
        if configured != original:
            torch._C._accelerator_setAllocatorSettings(original)


class B12xModelLoader(DefaultModelLoader):
    """Read safetensors with FlashInfer's direct checkpoint transport."""

    def __init__(self, load_config):
        options = load_config.model_loader_extra_config
        if not isinstance(options, dict):
            raise ValueError("b12x model_loader_extra_config must be a dict")
        options = options.copy()
        self.io_threads = options.pop("io_threads", 8)
        self.read_mode = options.pop("read_mode", "auto")
        if type(self.io_threads) is not int or not 1 <= self.io_threads <= 16:
            raise ValueError("io_threads must be an integer between 1 and 16")
        if self.read_mode not in ("auto", "bounce", "gds"):
            raise ValueError("read_mode must be auto, bounce or gds")
        if options.get("enable_multithread_load"):
            raise ValueError("b12x does not support enable_multithread_load")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError("b12x does not use safetensors_load_strategy")
        super().__init__(
            dataclasses.replace(load_config, model_loader_extra_config=options)
        )
        self._session = None
        self._file_filter = None

    def _resolve_mode(self, device, model_config):
        if device.type != "cuda":
            raise ValueError("the b12x loader requires a CUDA device")
        mode = self.read_mode
        if mode == "auto":
            from flashinfer.experimental.b12x.loader import capabilities

            index = (
                torch.accelerator.current_device_index()
                if device.index is None
                else device.index
            )
            mode = "bounce" if capabilities(index)["host_page_tables"] else "gds"
        if mode == "gds" and model_config.enable_cumem_allocator:
            raise ValueError("b12x shared GDS does not support the CuMem allocator")
        return mode

    def load_model(self, vllm_config, model_config, prefix=""):
        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        mode = self._resolve_mode(device, model_config)
        with _ipc_weight_allocations() if mode == "gds" else nullcontext():
            return super().load_model(vllm_config, model_config, prefix)

    @contextmanager
    def _open_session(self, model, model_config):
        config = get_current_vllm_config_or_none()
        parameter = next(model.parameters(), None)
        device = torch.device(
            self.load_config.device
            or (
                config.device_config.device
                if config
                else parameter.device
                if parameter is not None
                else "cuda"
            )
        )
        mode = self._resolve_mode(device, model_config)
        from flashinfer.experimental.b12x.loader._checkpoint import DirectWeightSession

        index = (
            torch.accelerator.current_device_index()
            if device.index is None
            else device.index
        )
        group = None
        if mode == "gds":
            from flashinfer.experimental.b12x.loader._shared_checkpoint import (
                SharedReadGroup,
            )

            from vllm.distributed.parallel_state import get_tp_group

            group = SharedReadGroup(get_tp_group().cpu_group, index)
        logger.info("Selected B12xModelLoader (O_DIRECT, read_mode=%s)", mode)
        with DirectWeightSession(
            index, io_threads=self.io_threads, shared_read_group=group, read_mode=mode
        ) as session:
            if group is not None:
                group.start(session)
            yield session

    def load_weights(self, model, model_config):
        if model_config.quantization == "torchao":
            raise ValueError("b12x does not support serialized torchao checkpoints")
        with (
            self._open_session(model, model_config) as session,
            weight_transfer(session),
        ):
            self._session = session
            self._file_filter = getattr(model, "checkpoint_file_weight_filter", None)
            try:
                super().load_weights(model, model_config)
                finish_weight_transfers()
                logger.info(
                    "B12X weight loading complete: %s", session.stats(flush=False)
                )
            finally:
                self._session = None
                self._file_filter = None

    @staticmethod
    def _needs_values(entry):
        return math.prod(entry.shape) == 1 or entry.name.rsplit(".", 1)[-1] in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }

    def _get_weights_iterator(self, source):
        session = self._session
        if session is None:
            raise RuntimeError("b12x requires an active checkpoint session")
        folder, files, _, index_file = self._prepare_source(source)
        if any(not path.endswith(".safetensors") for path in files):
            raise ValueError("b12x requires safetensors shards")
        index = Path(folder) / index_file
        weight_map = (
            json.loads(index.read_text())["weight_map"] if index.is_file() else None
        )
        indexed_paths = (
            {name: (index.parent / name).resolve() for name in set(weight_map.values())}
            if weight_map is not None
            else {}
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()

        def unused(name):
            return should_skip_weight(name, self.local_expert_ids) or (
                source.is_unused_weight is not None and source.is_unused_weight(name)
            )

        def ordinary(paths, skip):
            yield from session.weights(
                paths,
                prefix=source.prefix,
                index_path=index,
                needs_values=self._needs_values,
                skip=skip,
            )

        if self._file_filter is None:
            yield from ordinary(files, unused)
            return
        for path in files:
            resolved = Path(path).resolve()
            descriptors = [
                (source.prefix + name, file_source_tensor(descriptor))
                for name, descriptor in sorted(safetensors_file_sources(path).items())
                if self._file_filter(source.prefix + name)
                and not unused(name)
                and (
                    weight_map is None
                    or indexed_paths.get(weight_map.get(name)) == resolved
                )
            ]
            names = {name for name, _ in descriptors}
            yield from heapq.merge(
                descriptors,
                ordinary(
                    [path],
                    lambda name, names=names: (
                        unused(name) or source.prefix + name in names
                    ),
                ),
                key=lambda item: item[0],
            )
