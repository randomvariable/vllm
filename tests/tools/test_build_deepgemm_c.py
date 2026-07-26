# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from tools.build_deepgemm_C import build_command


def test_build_command_uses_cross_compilation_paths(monkeypatch):
    monkeypatch.setattr(
        "tools.build_deepgemm_C.torch.compiled_with_cxx11_abi", lambda: True
    )

    cmd = build_command(
        Path("/deepgemm"),
        Path("/out"),
        cxx="/usr/bin/aarch64-linux-gnu-g++",
        python_include="/target/python/include",
        ext_suffix=".cpython-312-aarch64-linux-gnu.so",
        cuda_home="/cuda",
        torch_root=Path("/target/torch"),
        cuda_lib_dir=Path("/cuda/targets/sbsa-linux/lib"),
        torch_cxx11_abi=0,
    )

    assert cmd[0] == "/usr/bin/aarch64-linux-gnu-g++"
    assert "-I/target/python/include" in cmd
    assert "-I/target/torch/include" in cmd
    assert "-I/target/torch/include/torch/csrc/api/include" in cmd
    assert "-L/target/torch/lib" in cmd
    assert "-L/cuda/targets/sbsa-linux/lib" in cmd
    assert "-D_GLIBCXX_USE_CXX11_ABI=0" in cmd
    assert str(Path("/out/_C.cpython-312-aarch64-linux-gnu.so")) == cmd[-1]


def test_build_command_preserves_native_torch_discovery(monkeypatch):
    monkeypatch.setattr(
        "tools.build_deepgemm_C.cpp_extension.include_paths",
        lambda **_: ["/native/torch/include"],
    )
    monkeypatch.setattr(
        "tools.build_deepgemm_C.cpp_extension.library_paths",
        lambda **_: ["/native/torch/lib", "/native/cuda/lib"],
    )
    monkeypatch.setattr(
        "tools.build_deepgemm_C.torch.compiled_with_cxx11_abi", lambda: False
    )

    cmd = build_command(
        Path("/deepgemm"),
        Path("/out"),
        cxx="g++",
        python_include="/native/python/include",
        ext_suffix=".cpython-312-x86_64-linux-gnu.so",
        cuda_home="/cuda",
    )

    assert "-I/native/torch/include" in cmd
    assert "-L/native/torch/lib" in cmd
    assert "-L/native/cuda/lib" in cmd
    assert "-L/cuda/lib64" in cmd
    assert "-D_GLIBCXX_USE_CXX11_ABI=0" in cmd
