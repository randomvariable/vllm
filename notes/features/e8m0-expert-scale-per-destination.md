# e8m0 expert scales: representation per destination parameter

Branch: `fix/e8m0-expert-scale-per-destination`
Upstream: [vllm-project/vllm#51946](https://github.com/vllm-project/vllm/pull/51946)
(fixes upstream issue #43416, "DeepSeek V4 Flash Model Output is Garbled")

## Problem

`DeepseekV4Model.load_weights` reinterprets *every* `float8_e8m0fnu` expert
scale in the checkpoint as raw `uint8` bytes, once per checkpoint tensor:

```python
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)
```

A ue8m0 byte `v` denotes `2 ** (v - 127)`. Which representation the weight
loader needs depends on the **destination parameter**, not on the checkpoint:

| Destination | Needs | Wrong representation does |
| --- | --- | --- |
| `uint8` (FP4 experts) | raw exponent bytes | numeric `copy_()` would map `2**-7` to `0` |
| `float32` block scale (`expert_dtype="fp8"`) | decoded value | writes the exponent *byte*: `2**-7` becomes `120.0` |

The second case scales every expert block by roughly `1e40` — silent garbage
output, no error.

Latent for our NVFP4/B12X deployments, which take the `uint8` path. It bites
any fp8-expert checkpoint.

## Fix

Decide per destination parameter inside the expert-mapping loop, after `param`
is known. `ue8m0_uint8_to_float` becomes a module-level function (it was a
static method on `DeepseekV4MegaMoEExperts`) so `dspark.py` can import it, and
the same per-destination choice is applied in the DSpark loader.

Decoding is exact: placing `v` in the IEEE-754 exponent field (bits 23..30)
reproduces `2 ** (v - 127)` with no rounding.

## Tests

`tests/models/deepseek_v4/test_e8m0_expert_scales.py`, hypothesis-driven over
exponent bytes:

- decode matches `2 ** (v - 127)` exactly for every representable exponent
- `uint8` destination receives the raw bytes unchanged
- float32 destination receives decoded values, never the byte
- the byte-as-float bug is pinned explicitly: `2**-7` must not arrive as `120.0`
- non-e8m0 scale tensors pass through untouched
