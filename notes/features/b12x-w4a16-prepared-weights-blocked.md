# B12x W4A16 prepared weights: blocked on unreleased FlashInfer API

Upstream: [vllm-project/vllm#52604](https://github.com/vllm-project/vllm/pull/52604)
Verdict: **cannot incorporate yet** — unmet external prerequisite, not a scope choice.

## What the PR does

Makes `FlashInferB12xExperts` prepare W4A16 weights during
`process_weights_after_loading()`, retain the caller-owned prepared object, and
execute through FlashInfer `run_prepared()` instead of a source-pointer cache.

## Why it cannot land here

**1. The FlashInfer API does not exist in any release.**
The two calls the top commit introduces —

```python
self._prepared_weights = self._wrapper.prepare_weights(...)
wrapper_output = wrapper.run_prepared(...)
```

come from [flashinfer-ai/flashinfer#4560](https://github.com/flashinfer-ai/flashinfer/pull/4560),
which is **open, unmerged** (checked via `gh api`: `state=open merged=false`).
Our pin is `flashinfer-python==0.6.17` (`requirements/cuda.txt`). The PR guards
neither call with `hasattr`/`getattr`, so porting it would raise `AttributeError`
in `process_weights_after_loading` for any W4A16 B12x MoE layer.

FlashInfer's own MoE design doc reserves the right to change `prepare_weights`
freely pre-release, so pinning a nightly to get it would be unstable.

**2. The PR is an explicitly temporary draft.**
Its description says only the top commit
(`13fafd778c2e5e7d06435838df82d7bf85312464`) is for review; the rest is
vLLM #43929, carried because GitHub cannot base an upstream PR on another
contributor's fork. It states the branch "will be rebased to a single commit
after #43929 merges".

**3. A config field is still missing.**
The commit reads `quant_config.source_format`, which this fork's
`FusedMoEQuantConfig` does not define:

| Field | Present in fork |
| --- | --- |
| `quant_dtype` | yes |
| `weight_quant_dtype` | yes |
| `a1_gscale` | yes |
| `source_format` | **no** |

It is supplied by the same in-flight upstream work.

## What the fork already has

The #43929 prerequisites did land through the rebase: `use_a16` plumbing and
`nvfp4_w4a16_moe_quant_config` routing are present in
`fused_moe/oracle/nvfp4.py` (lines 357, 369, 570, 586-588). So W4A16 B12x
routing works; only the explicit prepared-weight lifecycle is missing, and the
current path still recomputes MMA-layout scale views at load time
(`w1_sf_mma` / `w2_sf_mma`), which is functional.

## Revisit when

FlashInfer #4560 is merged and released, and vLLM #52604 has been rebased to
its single intended commit. At that point the port is the top commit plus the
`source_format` config field, against
`fused_moe/experts/flashinfer_b12x_moe.py` (which does exist here — it is
distinct from the fork-only native `fused_moe/b12x_moe.py`).
