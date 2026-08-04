# Model Resolution

vLLM loads HuggingFace-compatible models by inspecting the `architectures` field in `config.json` of the model repository
and finding the corresponding implementation that is registered to vLLM.
Nevertheless, our model resolution may fail for the following reasons:

- The `config.json` of the model repository lacks the `architectures` field.
- Unofficial repositories refer to a model using alternative names which are not recorded in vLLM.
- The same architecture name is used for multiple models, creating ambiguity as to which model should be loaded.

To fix this, explicitly specify the model architecture by passing `config.json` overrides to the `hf_overrides` option.
For example:

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

Our [list of supported models](../models/supported_models.md) shows the model architectures that are recognized by vLLM.

## How `hf_overrides` keys are applied

Multimodal checkpoints usually wrap the language model in a sub-config such as
`text_config`, and the model reads its hyperparameters from that sub-config rather
than from the top-level object. `hf_overrides` therefore routes each key to the
config that owns it:

- Dict-valued keys target the matching sub-config, and nested dicts are applied
  recursively:

    ```python
    hf_overrides={"text_config": {"num_experts_per_tok": 4}}
    ```

- Flat keys are applied to the config that already defines them. The top-level
  config is preferred, and the text config of a multimodal wrapper is used as a
  fallback. This means a flat key reaches the language model even when it only
  exists on the sub-config:

    ```python
    hf_overrides={"num_experts_per_tok": 4}
    ```

  Attributes that exist on both objects (such as `architectures`) resolve at the
  top level, and single-level text models are unaffected.

- Keys that match neither config are still applied to the top-level config, since
  some integrations inject their own keys, but they are logged as a warning. If an
  override appears to have no effect, check the logs for that warning first — it
  usually indicates a typo.
