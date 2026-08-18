# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's `_C` pybind11 extension for <TARGET_PY>.

Driven from cmake/external_projects/deepgemm.cmake. The driver runs against
the build interpreter's torch; <TARGET_PY> is only consulted for INCLUDEPY
and SOABI, so target venvs don't need torch installed.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR> <TARGET_PY>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

if len(sys.argv) != 4:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT> <TARGET_PY>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
target_py = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)

info = json.loads(
    subprocess.check_output(
        [
            target_py,
            "-c",
            "import sysconfig, json; "
            "print(json.dumps({k: sysconfig.get_config_var(k) "
            "for k in ('EXT_SUFFIX', 'INCLUDEPY')}))",
        ]
    ).decode()
)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    sys.exit("CUDA_HOME not found; cannot build DeepGEMM _C")

# Cross-compilation: the Makefile cross target sets these to the aarch64
# torch tree and SBSA CUDA library dir. When unset, fall back to the host
# torch/CUDA paths (native build).
torch_root = os.environ.get("DEEPGEMM_TORCH_ROOT")
cuda_lib_dir = os.environ.get("DEEPGEMM_CUDA_LIB_DIR")
torch_include = (
    [f"{torch_root}/include"]
    if torch_root
    else cpp_extension.include_paths(device_type="cuda")
)
torch_lib = (
    [f"{torch_root}/lib"]
    if torch_root
    else cpp_extension.library_paths(device_type="cuda")
)
cuda_lib = cuda_lib_dir or f"{cuda_home}/lib64"

# CCCL lives outside the standard CUDAToolkit search (mirrors DeepGEMM's setup.py).
includes = [
    info["INCLUDEPY"],
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    str(src / "csrc"),
    str(src / "deep_gemm/include"),
    str(src / "third-party/cutlass/include"),
    str(src / "third-party/cutlass/tools/util/include"),
    str(src / "third-party/fmt/include"),
    *torch_include,
]

cmd = [
    os.environ.get("CXX", "g++"),
    "-shared",
    "-fPIC",
    "-std=c++20",
    "-O3",
    "-g0",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    *(f"-I{p}" for p in includes),
    str(src / "csrc/python_api.cpp"),
    *(f"-L{p}" for p in torch_lib),
    f"-L{cuda_lib}",
    "-ltorch",
    "-ltorch_python",
    "-ltorch_cpu",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcudart",
    "-lnvrtc",
    "-o",
    str(out / f"_C{info['EXT_SUFFIX']}"),
]
print("[build_deepgemm_C] " + " ".join(cmd), flush=True)
subprocess.check_call(cmd)
