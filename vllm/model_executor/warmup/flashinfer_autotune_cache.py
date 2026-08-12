# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """Resolve the autotune cache file path.

    Pure path resolution: the containing directory is not created, so this is
    safe to call from every rank. Use `ensure_flashinfer_autotune_cache_dir`
    on the leader rank before writing.
    """
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    return output_dir / "autotune_configs.json"


def ensure_flashinfer_autotune_cache_dir(cache_path: Path) -> None:
    """Create the cache directory for `cache_path`.

    `exist_ok=True` still re-raises `FileExistsError` when the follow-up
    `is_dir()` probe misses, which happens on NFS-style shared storage with
    attribute caching, so tolerate that race explicitly.
    """
    with suppress(FileExistsError):
        cache_path.parent.mkdir(parents=True, exist_ok=True)


def prepare_flashinfer_autotune_cache_dir(
    world: "GroupCoordinator",
    cache_path: Path,
    *,
    is_leader: bool,
) -> None:
    """Create the cache directory on the leader and wait for the other ranks.

    Concurrent `mkdir` from every rank races on shared storage, so only the
    leader creates it and the barrier keeps followers from reading a directory
    that does not exist yet.
    """
    if is_leader:
        ensure_flashinfer_autotune_cache_dir(cache_path)
    if world.world_size > 1:
        world.barrier()


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    ensure_flashinfer_autotune_cache_dir(cache_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def publish_flashinfer_autotune_cache(
    world: "GroupCoordinator",
    cache_path: Path,
    contents: bytes,
    *,
    is_leader: bool,
) -> None:
    """Persist the broadcast autotune cache and synchronize all ranks.

    Only the leader writes, so ranks sharing storage do not duplicate write
    traffic. Ranks that do not share a filesystem with the leader materialize
    their own copy after the barrier so every rank can load the configs.
    """
    if is_leader:
        write_flashinfer_autotune_cache(cache_path, contents)
    world.barrier()
    if not is_leader and not cache_path.exists():
        write_flashinfer_autotune_cache(cache_path, contents)
