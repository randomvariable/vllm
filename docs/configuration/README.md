# Configuration Options

This section lists the most common options for running vLLM.

There are three main levels of configuration, from highest priority to lowest priority:

- [Request parameters](../serving/online_serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
- [Engine arguments](./engine_args.md)
- [Environment variables](./env_vars.md)

## Reasoning request controls

[ReSET entropy-threshold temperature](../features/reasoning_outputs.md#reset-entropy-threshold-temperature)
(`temperature_low`, `temperature_high`, `entropy_threshold`, `reset_window`)
is orthogonal to `thinking_token_budget`: the budget counts reasoning tokens
and forces `reasoning_end_str` when the limit is reached, deciding where the
reasoning block ends, while ReSET decides how tokens are sampled.
