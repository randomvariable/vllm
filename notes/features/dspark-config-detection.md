# DSpark checkpoint detection from config

Branch: `merge/dspark-config-detection`
Upstream: [vllm-project/vllm#52165](https://github.com/vllm-project/vllm/pull/52165)
(closes upstream issue #52111)

## Problem

`DeepSeek-V4-Flash-0731` and `DeepSeek-V4-Pro-0813` have no MTP heads — their
`mtp.*` tensors are DSpark drafters. Detection in `SpeculativeConfig.__post_init__`
is name- and architecture-based:

```python
elif (
    "dspark" in self.draft_model_config.model.lower()
    or "Qwen3DSparkModel" in self.draft_model_config.architectures
    or "Gemma4DSparkModel" in self.draft_model_config.architectures
    or ("DSparkDraftModel" in ... and hf_config.model_type == "qwen3")
):
```

Neither matches these checkpoints: the repo name has no `dspark`, and
`architectures` names the *target* model. They fall through to the
`model_type in MTPModelTypes` branch, get `method="mtp"`, route to
`DeepSeekV4MTPModel`, and die in the weight loader:

```text
KeyError: 'model.layers.43.mtp_block.main_norm.weight'   # Flash-0731
KeyError: 'model.layers.61.mtp_block.main_norm.weight'   # Pro-0813
```

Our serving config pins `method=dspark` explicitly, so this is latent for us —
it bites anyone who omits `--speculative-config`, and anyone who explicitly
asks for `method=mtp` gets the loader KeyError instead of a usable message.

## Discriminator

`dspark_target_layer_ids` is present only on DSpark checkpoints. Every
DeepSeek-V4 config declares `num_nextn_predict_layers`, so that key cannot
distinguish them.

`hf_config_override` rewrites `model_type` to `deepseek_mtp` and pins
`architectures = ["DeepSeekV4MTPModel"]`, so the predicate must accept both the
raw config shape and the already-overridden one.

## Scope kept out of the port

The upstream PR is against a much older base: a straight checkout would delete
fork-only `SpeculativeConfig` fields (`adaptive_speculative_tokens_window`,
`dspark_confidence_threshold`, `dspark_budget_frac`,
`dspark_capacity_verification_mode`). Only the additive predicates and their
call sites are taken.

The PR also replaces the architecture checks with a `DSparkVariant` enum used
for later dispatch. The fork's dispatch is already written against explicit
architecture strings and carries an extra `model_type == "qwen3"` guard on
`DSparkDraftModel` that upstream's enum drops; rewriting it would be a
behavioural change with no bug attached, so the guard is preserved and the enum
is not ported.

## Tests

`tests/config/test_dspark_draft_detection.py`, hypothesis-driven:

- DSpark keys ⇒ detected, for any repo name and either config shape
  (raw `deepseek_v4` / overridden `DeepSeekV4MTPModel`).
- Absent `dspark_target_layer_ids` ⇒ never detected from the DeepSeek-V4 path,
  whatever the other keys say.
- `DSparkDraftModel` keeps the fork's `model_type == "qwen3"` guard.
- Name fallback still works for arbitrary casing.
