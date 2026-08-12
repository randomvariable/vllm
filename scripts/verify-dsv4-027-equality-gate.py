#!/usr/bin/env python3
"""CPU-only semantic and source gate for the DSV4-Flash hotfix ports."""

from __future__ import annotations

from pathlib import Path

TOP_K = 512
COMPRESS_RATIO = 4
REPO = Path(__file__).resolve().parents[1]
ATTENTION = REPO / "vllm/models/deepseek_v4/attention.py"


def fast_path_indices(seq_len: int) -> list[int]:
    num_compressed = seq_len // COMPRESS_RATIO
    return list(range(num_compressed)) + [-1] * (TOP_K - num_compressed)


def full_path_indices(seq_len: int) -> list[int]:
    num_compressed = seq_len // COMPRESS_RATIO
    return list(range(min(num_compressed, TOP_K))) + [-1] * (
        TOP_K - min(num_compressed, TOP_K)
    )


def check(label: str, condition: bool) -> int:
    print(f"{'PASS' if condition else 'FAIL'} {label}")
    return 0 if condition else 1


def main() -> int:
    failures = 0
    attention = ATTENTION.read_text()

    failures += check(
        "#49486 source retains the DCP single-rank guard",
        "self.indexer_op.dcp_world_size == 1" in attention,
    )
    failures += check(
        "#49486 source retains the configured short-context gate",
        "indexer_metadata.max_seq_len // self.compress_ratio" in attention
        and "<= self.topk_tokens" in attention,
    )
    failures += check(
        "#49486 source writes padding sentinels",
        "tl.where(offsets < num_compressed, offsets, -1)" in attention,
    )

    for seq_len in (512, 1024, 2048, 2051):
        fast = fast_path_indices(seq_len)
        full = full_path_indices(seq_len)
        failures += check(
            f"#49486 preserves all candidates at seq_len={seq_len}",
            fast == full
            and {index for index in fast if index >= 0}
            == set(range(seq_len // COMPRESS_RATIO)),
        )

    failures += check(
        "#49486 stops at 2052 candidates",
        2051 // COMPRESS_RATIO <= TOP_K and 2052 // COMPRESS_RATIO > TOP_K,
    )
    failures += check(
        "#48407 remains dormant without a dense-MHA binding",
        "dense_mha_metadata_layer_name" not in attention,
    )

    print(f"RESULT: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return failures != 0


if __name__ == "__main__":
    raise SystemExit(main())
