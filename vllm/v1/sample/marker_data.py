# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical overthinking-marker lexical data and tokenizer resolution.

The overthinking-marker penalty (arXiv 2606.00206) suppresses the
leading-space branch-opening tokens that aggressive post-training
quantization amplifies in reasoning models. This module holds the exact
50-marker lexical list from the paper (Appendix C, Table 8) and the
tokenizer resolution that turns it into model-specific token IDs. It is
imported by both `ReasoningConfig` (model init and fail-closed request
validation) and the V2 sampler's marker-penalty state, so the marker set is
canonical in exactly one place.
"""

from __future__ import annotations

from vllm.logger import logger

# Canonical overthinking-marker set from arXiv 2606.00206, Appendix C,
# Table 8. All 50 items are leading-space variants ("_x" decodes to " x"): the
# paper penalizes the word-initial token, not the sentence-initial or subword
# variant of the same string. Tokens whose resolved ID collides with a bare
# high-frequency function word are rejected at resolution (see below).
OVERTHINKING_MARKERS: tuple[str, ...] = (
    " perhaps",
    " maybe",
    " wait",
    " Wait",
    " actually",
    " hold",
    " Hmm",
    " hmm",
    " Alternatively",
    " alternatively",
    " However",
    " however",
    " instead",
    " Instead",
    " But",
    " but",
    " though",
    " although",
    " yet",
    " rather",
    " unless",
    " otherwise",
    " nonetheless",
    " nevertheless",
    " regardless",
    " still",
    " anyway",
    " Or",
    " or",
    " either",
    " whether",
    " uncertain",
    " unsure",
    " possibly",
    " might",
    " could",
    " another",
    " different",
    " reconsider",
    " rethink",
    " backtrack",
    " retry",
    " recheck",
    " revisit",
    " doubt",
    " confused",
    " wrong",
    " mistake",
    " error",
    " incorrect",
)

# Bare high-frequency function words (with the leading space stripped) whose
# token IDs a marker must not collide with. The penalty operates on the *token
# ID*, so a marker that shares its ID with a common article or connective
# would penalize ordinary prose even when it is not used as a branch-opener.
# Any marker resolving to one of these is dropped at resolution.
_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "so",
        "if",
        "than",
        "that",
        "this",
        "it",
        "yes",
        "no",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "from",
        "up",
        "down",
        "out",
        "off",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "should",
        "shall",
        "not",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "then",
        "there",
        "here",
        "too",
        "very",
        "just",
    }
)


def resolve_marker_token_ids(tokenizer) -> list[int]:
    """Resolve the canonical overthinking markers to single-token IDs.

    Keeps only leading-space variants that tokenize to exactly one token,
    rejects any whose resolved ID decodes to a bare high-frequency function
    word, logs every unresolved / multi-token / collision item exactly once,
    and returns the sorted model-side marker set. The returned list is part of
    the calibration key for the reasoning-control stack.

    Args:
        tokenizer: The model's tokenizer, exposing ``encode(text,
            add_special_tokens=False)``, ``decode(token_ids,
            skip_special_tokens=False)`` and ``vocab_size``.

    Returns:
        Sorted list of resolved single-token marker IDs. May be empty; callers
        must fail closed when the feature is enabled but this list is empty.
    """
    resolved: list[int] = []
    for marker in OVERTHINKING_MARKERS:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        # The leading-space marker must resolve to exactly one token.
        if len(ids) != 1:
            logger.warning_once(
                "ReasoningMarkerPenalty: marker %r did not resolve to a "
                "single token (%d tokens: %r); skipped.",
                marker,
                len(ids),
                tuple(ids),
            )
            continue
        tid = int(ids[0])
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer-specific
            decoded = ""
        if decoded.lstrip().lower() in _FUNCTION_WORDS:
            logger.warning_once(
                "ReasoningMarkerPenalty: marker %r resolves to token %d which "
                "decodes to the high-frequency function word %r; rejected as "
                "a prose collision.",
                marker,
                tid,
                decoded,
            )
            continue
        if tid not in resolved:
            resolved.append(tid)
    resolved.sort()
    return resolved
