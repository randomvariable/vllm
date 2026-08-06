# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used. For e.g.,

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

## Online Examples

### Dynamic SD Eagle Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Telemetry

Dynamic SD currently reads its K values from a static, hand-written schedule. Choosing those numbers well requires knowing what acceptance the drafter actually achieves and what the target and draft forwards actually cost at each concurrency level. `--spec-decode-telemetry` measures both.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --spec-decode-telemetry ...
```

When enabled, the scheduler keeps, bucketed by batch size:

* an exponentially weighted acceptance rate, including per draft position, so a position that is never drafted reads as "no evidence" rather than as zero acceptance;
* exponentially weighted target and draft forward durations, measured with CUDA events in the worker rather than wall clock, because kernel launches and CUDA graph replays return before the GPU work completes.

Bucketing by batch size is what makes `K = 0` recoverable: nothing is drafted at `K = 0`, so acceptance there is unobservable, and a single global estimate would never learn its way back out. What was learned at `B = 8` survives `K = 0` at `B = 200`.

The drafter timer also runs at `K = 0`. The proposer still performs a cache-sync forward before returning an empty draft, so `K = 0` is cheaper than drafting but is not free.

### Cost

Default off. With the flag off, no CUDA events are allocated, nothing is recorded, and the worker reports no timings. With it on, the per-step cost is a fixed number of CUDA event records, a deferred read of events from two steps earlier (never a blocking wait), and O(K) arithmetic per drafting request, which is the same order as the acceptance counting that `SpecDecodingStats` already performs.

### Relationship to other flags

`--spec-decode-telemetry` is deliberately independent of `--disable-log-stats`. It is instrumentation for a future depth governor that will select K, not a reporting surface, so disabling statistics logging must not silently degrade control. It requires a speculative configuration; without one it allocates nothing.

Phase 1 records these signals and nothing more. K is still chosen entirely by `num_speculative_tokens_per_batch_size`, and enabling telemetry does not change which K is selected.

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value.
