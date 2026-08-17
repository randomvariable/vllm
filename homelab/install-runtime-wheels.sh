#!/bin/sh
set -eu

PYTHON=/opt/venv/bin/python
uv pip install --python "$PYTHON" -r /runtime-requirements/cuda.txt
uv pip install --python "$PYTHON" --no-deps /wheels-b12x/*.whl

if ls /wheels-flashinfer/flashinfer_cubin-*.whl >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" --no-index \
        --find-links /wheels-flashinfer \
        /wheels-flashinfer/flashinfer_cubin-*.whl
fi
if ls /wheels-flashinfer/flashinfer_python-*.whl >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" \
        /wheels-flashinfer/flashinfer_python-*.whl
fi
if ls /wheels-flashinfer/flashinfer_jit_cache-*.whl >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" --no-deps \
        /wheels-flashinfer/flashinfer_jit_cache-*.whl
fi

uv pip install --python "$PYTHON" xxhash
uv pip install --python "$PYTHON" --no-deps /wheels/*.whl

# nvidia-cutlass-dsl aarch64 wheel bug: the Python frontend serializes
# `enable-pyir=false` into the cute-to-nvvm pass pipeline, but the compiled
# native pass does not register that option, so PassManager.parse() rejects the
# whole pipeline ("failed to add cute-to-nvvm ... no such option enable-pyir"),
# which surfaces as an ICE on every b12x cute.compile() call. `enable-pyir`
# defaults false and b12x kernels use the standard preprocessor, so dropping
# the token is safe; the pass compiles fine without it.
"$PYTHON" - <<'PYEOF'
import pathlib
p = pathlib.Path("/opt/venv/lib/python3.13/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/compiler.py")
s = p.read_text()
# BooleanCompileOption.serialize always emits `enable-pyir=false`, but the
# native cute-to-nvvm pass does not register that option, so PassManager.parse
# rejects the pipeline and surfaces as an ICE. Drop the token when false
# (default; b12x uses the standard preprocessor, never the PyIR frontend).
old = 'return f"{self.__class__._option_name}={\'true\' if self._value else \'false\'}"'
new = ('if self.__class__._option_name == "enable-pyir" and not self._value:\n'
       '            return ""\n'
       '        ' + old)
if old in s:
    p.write_text(s.replace(old, new, 1))
    print("patched: BooleanCompileOption.serialize drops enable-pyir when false")
else:
    raise SystemExit(f"patch target not found in {p}")
PYEOF

rm -rf /wheels /wheels-b12x /wheels-flashinfer
