# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from shutil import which

try:
    # Try to get CUDA_HOME from PyTorch installation, which is the
    # most reliable source of truth for vLLM's build.
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    print("Warning: PyTorch not found. Falling back to CUDA_HOME environment variable.")
    CUDA_HOME = os.environ.get("CUDA_HOME")


def get_python_executable():
    """Get the current Python executable, which is used to run this script."""
    return sys.executable


def get_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def _torch_hip_version():
    """Get the HIP version PyTorch was built against, if any.

    Returns:
        The HIP version string when the installed PyTorch is a ROCm build,
        otherwise ``None`` (including when PyTorch is not importable).
    """
    try:
        import torch

        return torch.version.hip
    except ImportError:
        return None


def detect_backend():
    """Determine whether presets should target CUDA or ROCm.

    Detection mirrors ``setup.py``: an explicit ``VLLM_TARGET_DEVICE`` wins,
    otherwise a ROCm PyTorch build selects ROCm. Hosts where neither toolchain
    announces itself fall back to CUDA, preserving this script's original
    behaviour.

    Returns:
        Either ``"cuda"`` or ``"rocm"``.
    """
    target_device = os.environ.get("VLLM_TARGET_DEVICE")
    if target_device == "rocm":
        print("VLLM_TARGET_DEVICE=rocm: generating ROCm presets.")
        return "rocm"
    if target_device == "cuda":
        return "cuda"

    if _torch_hip_version():
        print("Auto-detected ROCm: the installed PyTorch is a ROCm build.")
        return "rocm"

    return "cuda"


def find_rocm_root():
    """Locate the ROCm installation prefix.

    Probes, in order, the ROCm SDK Python wheels (``rocm-sdk path --root``),
    the conventional ROCm environment variables, and finally ``/opt/rocm``.

    Returns:
        The ROCm root directory, or ``None`` when no candidate is found.
    """
    rocm_sdk = which("rocm-sdk")
    if rocm_sdk:
        try:
            root = subprocess.run(
                [rocm_sdk, "path", "--root"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            root = ""
        if root and os.path.isdir(root):
            print(f"Found ROCm root via 'rocm-sdk path --root': {root}")
            return root

    for env_var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(env_var)
        if root and os.path.isdir(root):
            print(f"Found ROCm root via {env_var}: {root}")
            return root

    if os.path.isdir("/opt/rocm"):
        print("Found ROCm root at /opt/rocm")
        return "/opt/rocm"

    return None


def find_hip_device_lib_path(rocm_root):
    """Find the ROCm device bitcode directory beneath a ROCm root.

    Args:
        rocm_root: The ROCm installation prefix to search.

    Returns:
        The first directory named ``bitcode`` under ``rocm_root``, or ``None``
        when the installation does not ship one at a discoverable location.
    """
    for dirpath, dirnames, _ in os.walk(rocm_root):
        if "bitcode" in dirnames:
            return os.path.join(dirpath, "bitcode")
    return None


def detect_gpu_arch():
    """Determine the AMD GPU architecture to build kernels for.

    Honours ``PYTORCH_ROCM_ARCH`` when set, otherwise queries the attached
    device through PyTorch. Prompts only when neither source answers.

    Returns:
        A gfx architecture string such as ``gfx1151``.

    Raises:
        ValueError: If the architecture cannot be determined.
    """
    arch = os.environ.get("PYTORCH_ROCM_ARCH")
    if arch:
        print(f"Using AMD GPU architecture from PYTORCH_ROCM_ARCH: {arch}")
        return arch

    try:
        import torch

        if torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            if arch:
                print(f"Found AMD GPU architecture via PyTorch: {arch}")
                return arch
    except (ImportError, AssertionError, RuntimeError, IndexError):
        pass

    arch = input(
        "Could not automatically detect the AMD GPU architecture. Please "
        "provide the gfx target to build for (e.g., gfx1151): "
    ).strip()
    if not arch:
        raise ValueError(
            "Could not determine the AMD GPU architecture. Please provide it "
            "manually or set PYTORCH_ROCM_ARCH."
        )
    return arch


def detect_nvcc():
    """Locate the CUDA compiler, prompting when it cannot be found.

    Returns:
        A dict with the detected ``nvcc_path``.
    """
    nvcc_path = None
    if CUDA_HOME:
        prospective_path = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.exists(prospective_path):
            nvcc_path = prospective_path
            print(f"Found nvcc via torch.utils.cpp_extension.CUDA_HOME: {nvcc_path}")

    if not nvcc_path:
        nvcc_path = which("nvcc")
        if nvcc_path:
            print(f"Found nvcc in PATH: {nvcc_path}")

    if not nvcc_path:
        nvcc_path_input = input(
            "Could not automatically find 'nvcc'. Please provide the full "
            "path to nvcc (e.g., /usr/local/cuda/bin/nvcc): "
        )
        nvcc_path = nvcc_path_input.strip()
    print(f"Using NVCC path: {nvcc_path}")
    return {"nvcc_path": nvcc_path}


def detect_rocm_toolchain():
    """Locate the ROCm toolchain, prompting when it cannot be found.

    vLLM's HIP extensions must be compiled with the ``amdclang`` toolchain that
    ships inside ROCm. Building the host side with gcc mixes C++ ABIs against
    the PyTorch ROCm wheels and segfaults at import time, so the compilers are
    pinned here rather than left to CMake's default probe.

    Returns:
        A dict with the detected ``rocm_root``, ``c_compiler``,
        ``cxx_compiler`` and ``gpu_arch``.

    Raises:
        ValueError: If no ROCm installation can be located.
    """
    rocm_root = find_rocm_root()
    if not rocm_root:
        rocm_root = input(
            "Could not automatically find a ROCm installation. Please provide "
            "the full path to the ROCm root (e.g., /opt/rocm): "
        ).strip()
        if not rocm_root:
            raise ValueError(
                "Could not determine the ROCm root. Please provide it manually "
                "or set ROCM_PATH."
            )
    print(f"Using ROCm root: {rocm_root}")

    llvm_bin = os.path.join(rocm_root, "lib", "llvm", "bin")
    c_compiler = os.path.join(llvm_bin, "amdclang")
    cxx_compiler = os.path.join(llvm_bin, "amdclang++")
    if not os.path.exists(c_compiler):
        print(
            f"Warning: '{c_compiler}' not found. vLLM's extensions must be built "
            "with amdclang to stay ABI-compatible with the PyTorch ROCm wheels; "
            "check that the ROCm root is complete."
        )
    print(f"Using HIP compilers: {c_compiler} and {cxx_compiler}")

    return {
        "rocm_root": rocm_root,
        "c_compiler": c_compiler,
        "cxx_compiler": cxx_compiler,
        "gpu_arch": detect_gpu_arch(),
    }


def generate_presets(output_path="CMakeUserPresets.json", force_overwrite=False):
    """Generates the CMakeUserPresets.json file."""

    print("Attempting to detect your system configuration...")

    backend = detect_backend()

    # Detect the GPU toolchain
    toolchain = detect_rocm_toolchain() if backend == "rocm" else detect_nvcc()

    # Detect Python executable
    python_executable = get_python_executable()
    if python_executable:
        print(f"Found Python via sys.executable: {python_executable}")
    else:
        python_executable_prompt = (
            "Could not automatically find Python executable. Please provide "
            "the full path to your Python executable for vLLM development "
            "(typically from your virtual environment, e.g., "
            "/home/user/venvs/vllm/bin/python): "
        )
        python_executable = input(python_executable_prompt).strip()
        if not python_executable:
            raise ValueError(
                "Could not determine Python executable. Please provide it manually."
            )

    print(f"Using Python executable: {python_executable}")

    # Get CPU cores
    cpu_cores = get_cpu_cores()
    if backend == "rocm":
        # hipcc has no nvcc-style intra-file threading, so every core is
        # available to the job pool.
        nvcc_threads = 0
        cmake_jobs = max(1, cpu_cores)
        print(f"Detected {cpu_cores} CPU cores. Setting CMake jobs={cmake_jobs}.")
    else:
        nvcc_threads = min(4, cpu_cores)
        cmake_jobs = max(1, cpu_cores // nvcc_threads)
        print(
            f"Detected {cpu_cores} CPU cores. "
            f"Setting NVCC_THREADS={nvcc_threads} and CMake jobs={cmake_jobs}."
        )

    # Get vLLM project root (assuming this script is in vllm/tools/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"VLLM project root detected as: {project_root}")

    # Ensure python_executable path is absolute or resolvable
    if not os.path.isabs(python_executable) and which(python_executable):
        python_executable = os.path.abspath(which(python_executable))
    elif not os.path.isabs(python_executable):
        print(
            f"Warning: Python executable '{python_executable}' is not an "
            "absolute path and not found in PATH. CMake might not find it."
        )

    environment = {}
    if backend == "rocm":
        rocm_root = toolchain["rocm_root"]
        cxx_compiler = toolchain["cxx_compiler"]
        gpu_arch = toolchain["gpu_arch"]
        llvm_bin = os.path.join(rocm_root, "lib", "llvm", "bin")
        cache_variables = {
            "CMAKE_C_COMPILER": toolchain["c_compiler"],
            "CMAKE_CXX_COMPILER": cxx_compiler,
            "CMAKE_HIP_COMPILER": cxx_compiler,
            "CMAKE_BUILD_TYPE": "Release",
            "VLLM_PYTHON_EXECUTABLE": python_executable,
            "VLLM_TARGET_DEVICE": "rocm",
            "CMAKE_INSTALL_PREFIX": "${sourceDir}",
            "CMAKE_PREFIX_PATH": os.path.join(rocm_root, "lib", "cmake"),
            "CMAKE_HIP_FLAGS": "",
            "ROCM_PATH": rocm_root,
            "HIP_PATH": rocm_root,
            "AMDGPU_TARGETS": gpu_arch,
            "HIP_ARCHITECTURES": gpu_arch,
        }
        environment = {
            "ROCM_PATH": rocm_root,
            "HIP_PATH": rocm_root,
            "ROCM_HOME": rocm_root,
            "PATH": f"{os.path.join(rocm_root, 'bin')}:{llvm_bin}:$penv{{PATH}}",
        }
        device_lib_path = find_hip_device_lib_path(rocm_root)
        if device_lib_path:
            print(f"Found HIP device bitcode: {device_lib_path}")
            environment["HIP_DEVICE_LIB_PATH"] = device_lib_path
        else:
            print(
                "Warning: no 'bitcode' directory found under the ROCm root. "
                "HIP_DEVICE_LIB_PATH was not set; device compilation may fail."
            )
    else:
        cache_variables = {
            "CMAKE_CUDA_COMPILER": toolchain["nvcc_path"],
            "CMAKE_BUILD_TYPE": "Release",
            "VLLM_PYTHON_EXECUTABLE": python_executable,
            "CMAKE_INSTALL_PREFIX": "${sourceDir}",
            "CMAKE_CUDA_FLAGS": "",
            "NVCC_THREADS": str(nvcc_threads),
        }

    # Detect compiler cache
    if which("sccache"):
        print("Using sccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "sccache"
    elif which("ccache"):
        print("Using ccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "ccache"
    else:
        print("No compiler cache ('ccache' or 'sccache') found.")

    configure_preset = {
        "name": "release",
        "binaryDir": "${sourceDir}/cmake-build-release",
        "cacheVariables": cache_variables,
    }
    if environment:
        configure_preset["environment"] = environment
    if which("ninja"):
        print("Using Ninja generator.")
        configure_preset["generator"] = "Ninja"
        cache_variables["CMAKE_JOB_POOLS"] = f"compile={cmake_jobs}"
    else:
        print("Ninja not found, using default generator. Build may be slower.")

    presets = {
        "version": 6,
        # Keep in sync with CMakeLists.txt, requirements/build/cuda.txt and
        # requirements/build/rocm.txt
        "cmakeMinimumRequired": {"major": 3, "minor": 26, "patch": 1},
        "configurePresets": [configure_preset],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
                "jobs": cmake_jobs,
            }
        ],
    }

    output_file_path = os.path.join(project_root, output_path)

    if os.path.exists(output_file_path):
        if force_overwrite:
            print(f"Overwriting existing file '{output_file_path}'")
        else:
            overwrite = (
                input(f"'{output_file_path}' already exists. Overwrite? (y/N): ")
                .strip()
                .lower()
            )
            if overwrite != "y":
                print("Generation cancelled.")
                return

    try:
        with open(output_file_path, "w") as f:
            json.dump(presets, f, indent=4)
        print(f"Successfully generated '{output_file_path}'")
        print("\nTo use this preset:")
        print(f"1. Ensure you are in the vLLM root directory: cd {project_root}")
        print("2. Initialize CMake: cmake --preset release")
        print("3. Build+install: cmake --build --preset release --target install")

    except OSError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite existing CMakeUserPresets.json without prompting",
    )

    args = parser.parse_args()
    generate_presets(force_overwrite=args.force_overwrite)
