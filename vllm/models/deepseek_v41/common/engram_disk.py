# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram table rows read on the host from the checkpoint files."""

import mmap
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.models.qwen4_exp.nvidia.ngram_embedding import _maps_entry

# Parameter names of a table's planes: FP8 or packed MXFP4 values, then E8M0
# block scales.
PLANES = ("weight", "weight_scale_inv")


def _read_rows(fd: int, base: int, ids: torch.Tensor, rows: torch.Tensor) -> None:
    """Read the rows of sorted distinct `ids` of the table at `base` in `fd`."""
    page, row_bytes = mmap.PAGESIZE, rows.shape[1]
    start = base + ids * row_bytes
    first, last = start // page, (start + row_bytes - 1) // page
    run = torch.ones_like(first, dtype=torch.bool)
    run[1:] = first[1:] > last[:-1] + 1
    # Queue every page run before the serial reads.
    for lo, hi in zip(first[run].tolist(), last[run.roll(-1)].tolist()):
        os.posix_fadvise(fd, lo * page, (hi - lo + 1) * page, os.POSIX_FADV_WILLNEED)
    for row, offset in zip(rows.numpy(), start.tolist()):
        if os.preadv(fd, [row], offset) != row_bytes:
            raise ValueError(
                f"Engram table file {os.readlink(f'/proc/self/fd/{fd}')} is "
                f"truncated at offset {offset}"
            )


class EngramDiskRows:
    """One rank's rows of an Engram table, read from the checkpoint files.

    The loader's zero-copy views of the safetensors files only locate the
    rows: `bind` records each plane's file and offset and drops the views, so
    the table takes no device or pinned host memory. Each step, `start`
    copies the rank's hash ids to the host, and a worker thread reads the
    distinct rows they select with `preadv` into pinned staging while the GPU
    runs the layers before the Engram layer. `finish` uploads that compact
    table with the ids renumbered into it (`vocab_start` + its row), so the
    lookup kernel reads it like the resident table.
    """

    def __init__(
        self,
        plane_bytes: tuple[int, int],
        weight_dtype: torch.dtype,
        vocab: tuple[int, int],
        heads: tuple[int, int],
        num_heads: int,
        max_tokens: int,
        dummy: bool,
    ) -> None:
        """`vocab`: the rank's [start, end) table rows; `heads`: its
        [start, end) hash heads out of `num_heads`; `dummy`: weights are
        not loaded, so rows read as zeros."""
        self.vocab, self.heads, self._dummy = vocab, heads, dummy
        self._weight_dtype = weight_dtype
        device = torch.get_default_device()
        # Placeholder (meta) modules never run, so allocate nothing real.
        host = device if device.type == "meta" else torch.device("cpu")
        pin = device.type == "cuda"
        max_rows = max_tokens * (heads[1] - heads[0])

        def staging(*shape: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
            return (
                torch.empty(shape, dtype=dtype, device=host, pin_memory=pin),
                torch.empty(shape, dtype=dtype, device=device),
            )

        self._planes = [
            staging(max_rows, size, dtype=torch.uint8) for size in plane_bytes
        ]
        self._host_ids = torch.empty(
            max_tokens,
            heads[1] - heads[0],
            dtype=torch.int32,
            device=host,
            pin_memory=pin,
        )
        # Columns of other ranks' heads stay -1: no row.
        self._ids = staging(max_tokens, num_heads, dtype=torch.int32)
        self._ids[0].fill_(-1)
        self._copied = torch.cuda.Event() if pin else None
        self._uploaded = torch.cuda.Event() if pin else None
        self._has_upload = False
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="engram-disk")
        self._pending: Future[int] | None = None
        self._num_tokens = 0
        self._views: dict[str, torch.Tensor] = {}
        self._sources: list[tuple[int, int]] = []
        self._fds: dict[str, int] = {}
        self._bound = False

    def __del__(self) -> None:
        for fd in getattr(self, "_fds", {}).values():
            os.close(fd)

    def keep_view(self, plane: str, loaded_weight: torch.Tensor) -> None:
        """Keep the rank's rows of a checkpoint table as a zero-copy view."""
        if self._bound:
            raise RuntimeError("Engram disk offload does not support weight reload")
        if plane in self._views:
            raise ValueError(f"Duplicate Engram {plane} checkpoint tensor")
        if loaded_weight.device.type != "cpu" or not loaded_weight.is_contiguous():
            raise ValueError("Engram disk offload requires contiguous CPU mmap views")
        if plane == PLANES[1] and loaded_weight.dtype not in (
            torch.uint8,
            torch.float8_e8m0fnu,
        ):
            raise ValueError(
                f"Engram scales must be E8M0 or uint8, got {loaded_weight.dtype}"
            )
        if plane == PLANES[0] and loaded_weight.dtype != self._weight_dtype:
            raise ValueError(
                f"Engram {plane} dtype {loaded_weight.dtype} must match the table "
                f"dtype {self._weight_dtype}; disk offload serves rows as stored"
            )
        if plane == PLANES[1] and loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        start, end = self.vocab
        rows = loaded_weight.narrow(0, start, end - start).view(torch.uint8)
        width = self._planes[PLANES.index(plane)][0].shape[1]
        if rows.shape != (end - start, width):
            raise ValueError(
                f"Engram {plane} rows {tuple(rows.shape)} do not fit "
                f"({end - start}, {width})"
            )
        self._views[plane] = rows

    def bind(self) -> None:
        """Record each plane's file and offset and release the loader's views."""
        if self._bound or self._dummy:
            self._bound = True
            return
        for plane in PLANES:
            view = self._views.get(plane)
            if view is None:
                raise ValueError(f"Engram {plane} was not loaded from the checkpoint")
            # The loader hands over mmap views, so find their files here.
            path, offset = _maps_entry(view.data_ptr())
            if not path or not os.path.isfile(path):
                raise ValueError(
                    f"Engram {plane} is not a file-backed view "
                    f"({path or 'anonymous'}); disk offload reads the "
                    "safetensors files the loader maps"
                )
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            self._sources.append((self._fds[path], offset))
        self._views.clear()
        self._bound = True

    @property
    def tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The values and scales staged on the device for the last step."""
        values, scales = (device for _, device in self._planes)
        return values.view(self._weight_dtype), scales

    def start(self, hash_ids: torch.Tensor) -> None:
        """Read the rows of `hash_ids` ([tokens, heads]) on the worker."""
        if self._pending is not None:
            # A step that never looked its rows up.
            self._pending.result()
        if not self._bound:
            raise RuntimeError("Engram disk table has not been bound")
        num_tokens = hash_ids.shape[0]
        self._host_ids[:num_tokens].copy_(
            hash_ids[:, self.heads[0] : self.heads[1]], non_blocking=True
        )
        if self._copied is not None:
            self._copied.record()
        self._num_tokens = num_tokens
        self._pending = self._worker.submit(self._read, num_tokens)

    def _read(self, num_tokens: int) -> int:
        if self._copied is not None:
            self._copied.synchronize()
        if self._has_upload:
            assert self._uploaded is not None
            self._uploaded.synchronize()
        ids = self._host_ids[:num_tokens]
        start, end = self.vocab
        owned = (ids >= start) & (ids < end)
        rows, inverse = (ids[owned].long() - start).unique(return_inverse=True)
        num_rows = rows.numel()
        if self._sources:
            for (fd, base), (host, _) in zip(self._sources, self._planes):
                _read_rows(fd, base, rows, host[:num_rows])
        else:
            for host, _ in self._planes:
                host[:num_rows].zero_()
        compact = self._ids[0][:num_tokens, self.heads[0] : self.heads[1]]
        compact.fill_(-1)
        compact[owned] = inverse.int() + start
        return num_rows

    def finish(self) -> torch.Tensor:
        """Upload the rows read since `start`; return the renumbered ids."""
        assert self._pending is not None, "Engram disk rows were not started"
        num_rows = self._pending.result()
        self._pending = None
        for host, device in self._planes:
            device[:num_rows].copy_(host[:num_rows], non_blocking=True)
        host, device = self._ids
        ids = device[: self._num_tokens]
        ids.copy_(host[: self._num_tokens], non_blocking=True)
        if self._uploaded is not None:
            self._uploaded.record()
            self._has_upload = True
        return ids
