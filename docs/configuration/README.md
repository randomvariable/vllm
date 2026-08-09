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

Configure markers with `reasoning_marker_strs` in `--reasoning-config`. A
marker may span several tokens, such as `"let me think"`. Multi-token markers
are penalised only where they would complete: the preceding tokens must
already match the generated text, and the penalty applies to the marker's
final token alone, so a shared prefix is not discouraged on its own. The
penalty applies only between the configured `reasoning_start_str` and
`reasoning_end_str`, not to the final answer content.

Reasoning controls require an enabled reasoning configuration. Start the
server with `--reasoning-parser` and/or `--reasoning-config` so vLLM can
initialize the reasoning boundary tokens, for example:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "reasoning_marker_strs": ["Wait", "let me think"]}'
```

Pass the penalty in an OpenAI-compatible request through `extra_body`:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"reasoning_marker_penalty": 0.5},
)
```

Under speculative decoding, speculative-token rows are included in penalty
application, while marker-token and reasoning-block semantics remain
unchanged.

`reasoning_marker_penalty`, `thinking_token_budget` and
[`reasoning_answer_reserve`](../features/reasoning_outputs.md#answer-reserve)
are independent and can be set together. The penalty discourages selected
tokens; the budget counts reasoning tokens separately and forces
`reasoning_end_str` when the limit is reached; the reserve forces
`reasoning_end_str` once the remaining output budget drops to it.

[Step-aware temperature](../features/reasoning_outputs.md#step-aware-temperature)
(`temperature_final`, `temperature_anneal_steps` and
`reasoning_answer_temperature`) is orthogonal to all three: those parameters
decide where the reasoning block ends, while step-aware temperature decides how
tokens are sampled on either side of it.
