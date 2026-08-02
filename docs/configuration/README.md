# Configuration Options

This section lists the most common options for running vLLM.

There are three main levels of configuration, from highest priority to lowest priority:

- [Request parameters](../serving/online_serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
- [Engine arguments](./engine_args.md)
- [Environment variables](./env_vars.md)

## Reasoning request controls

`reasoning_marker_penalty` is an optional per-request sampling parameter for
reasoning models. It subtracts the specified penalty from logits for configured
reasoning marker tokens, discouraging those tokens while generation is inside a
reasoning block. The value must be finite and non-negative; `0` or an omitted
value disables the penalty.

Configure markers through `reasoning_marker_strs` in `--reasoning-config`. Each
marker must encode as exactly one tokenizer token. Markers that encode as
multiple tokens are ignored with a warning. The penalty applies only between
the configured `reasoning_start_str` and `reasoning_end_str`, not to final
answer content.

Reasoning controls require an enabled reasoning configuration. Start the
server with `--reasoning-parser` and/or `--reasoning-config` so vLLM can
initialize the reasoning boundary tokens, for example:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "reasoning_marker_strs": ["."]}'
```

Pass the penalty in an OpenAI-compatible request through `extra_body`:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"reasoning_marker_penalty": 0.5},
)
```

The control is not supported by the V2 model runner; use
`VLLM_USE_V2_MODEL_RUNNER=0`. With speculative decoding, speculative-token
rows are included in penalty application, while marker-token and
reasoning-block semantics remain unchanged.

`reasoning_marker_penalty` and `thinking_token_budget` are independent and can
be set together. The penalty discourages selected tokens; the budget counts
reasoning tokens separately and forces `reasoning_end_str` when its limit is
reached.
