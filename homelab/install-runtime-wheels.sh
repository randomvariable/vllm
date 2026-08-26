#!/bin/sh
set -eu

PYTHON=/opt/venv/bin/python
uv pip install --python "$PYTHON" -r /runtime-requirements/cuda.txt
uv pip install --python "$PYTHON" -r /runtime-requirements/kv_connectors.txt

KV_METADATA=$("$PYTHON" - <<'PYEOF'
import importlib.metadata as metadata

import torch

cuda_version = torch.version.cuda
if cuda_version is None:
    raise SystemExit("torch.version.cuda is not set")

print(
    cuda_version.split(".", 1)[0],
    metadata.version("nixl"),
    metadata.version("mooncake-transfer-engine"),
)
PYEOF
)
IFS=' ' read -r CUDA_MAJOR NIXL_VERSION MOONCAKE_VERSION <<EOF
$KV_METADATA
EOF

uv pip uninstall --python "$PYTHON" nixl-cu12 nixl-cu13 2>/dev/null || true
uv pip install --python "$PYTHON" --no-deps "nixl-cu${CUDA_MAJOR}==${NIXL_VERSION}"

if [ "$CUDA_MAJOR" = 13 ]; then
    uv pip uninstall --python "$PYTHON" mooncake-transfer-engine
    uv pip install --python "$PYTHON" \
        "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi

"$PYTHON" - <<'PYEOF'
import importlib.metadata as metadata

for package in ("lmcache", "mooncake-transfer-engine-cuda13", "nixl-cu13"):
    print(f"{package}=={metadata.version(package)}")
PYEOF
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
import pathlib, sysconfig
p = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/compiler.py"
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
