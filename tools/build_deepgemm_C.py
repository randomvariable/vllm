# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's ``_C`` pybind11 extension for a target Python."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.utils import cpp_extension


def _env_default(name: str) -> str | None:
    return os.environ.get(f"DEEPGEMM_{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("target_python")
    parser.add_argument("--cxx", default=_env_default("CXX"))
    parser.add_argument(
        "--python-include", default=_env_default("PYTHON_INCLUDE")
    )
    parser.add_argument("--ext-suffix", default=_env_default("EXT_SUFFIX"))
    parser.add_argument("--torch-root", type=Path, default=_env_default("TORCH_ROOT"))
    parser.add_argument(
        "--cuda-lib-dir", type=Path, default=_env_default("CUDA_LIB_DIR")
    )
    parser.add_argument(
        "--torch-cxx11-abi",
        type=int,
        choices=(0, 1),
        default=_env_default("TORCH_CXX11_ABI"),
    )
    return parser.parse_args()


def target_python_info(target_python: str) -> dict[str, str]:
    query = (
        "import sysconfig, json; "
        "print(json.dumps({k: sysconfig.get_config_var(k) "
        "for k in ('EXT_SUFFIX', 'INCLUDEPY')}))"
    )
    return json.loads(
        subprocess.check_output(
            [
                target_python,
                "-c",
                query,
            ]
        ).decode()
    )


def build_command(
    src: Path,
    out: Path,
    *,
    cxx: str,
    python_include: str,
    ext_suffix: str,
    cuda_home: str,
    torch_root: Path | None = None,
    cuda_lib_dir: Path | None = None,
    torch_cxx11_abi: int | None = None,
) -> list[str]:
    if torch_root is None:
        torch_includes = cpp_extension.include_paths(device_type="cuda")
        torch_library_dirs = cpp_extension.library_paths(device_type="cuda")
    else:
        torch_includes = [
            str(torch_root / "include"),
            str(torch_root / "include/torch/csrc/api/include"),
        ]
        torch_library_dirs = [str(torch_root / "lib")]

    includes = [
        python_include,
        f"{cuda_home}/include",
        f"{cuda_home}/include/cccl",
        str(src / "csrc"),
        str(src / "deep_gemm/include"),
        str(src / "third-party/cutlass/include"),
        str(src / "third-party/cutlass/tools/util/include"),
        str(src / "third-party/fmt/include"),
        *torch_includes,
    ]
    cuda_lib_dir = cuda_lib_dir or Path(cuda_home) / "lib64"
    if torch_cxx11_abi is None:
        torch_cxx11_abi = int(torch.compiled_with_cxx11_abi())

    return [
        cxx,
        "-shared",
        "-fPIC",
        "-std=c++20",
        "-O3",
        "-g0",
        "-Wno-psabi",
        "-Wno-deprecated-declarations",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_C",
        f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}",
        *(f"-I{path}" for path in includes),
        str(src / "csrc/python_api.cpp"),
        *(f"-L{path}" for path in torch_library_dirs),
        f"-L{cuda_lib_dir}",
        "-ltorch",
        "-ltorch_python",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        "-lc10_cuda",
        "-lcudart",
        "-lnvrtc",
        "-o",
        str(out / f"_C{ext_suffix}"),
    ]


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    info = (
        target_python_info(args.target_python)
        if args.python_include is None or args.ext_suffix is None
        else {}
    )
    python_include = args.python_include or info["INCLUDEPY"]
    ext_suffix = args.ext_suffix or info["EXT_SUFFIX"]
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise SystemExit("CUDA_HOME not found; cannot build DeepGEMM _C")

    cmd = build_command(
        src,
        out,
        cxx=args.cxx or os.environ.get("CXX", "g++"),
        python_include=python_include,
        ext_suffix=ext_suffix,
        cuda_home=cuda_home,
        torch_root=args.torch_root,
        cuda_lib_dir=args.cuda_lib_dir,
        torch_cxx11_abi=args.torch_cxx11_abi,
    )
    print("[build_deepgemm_C] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
