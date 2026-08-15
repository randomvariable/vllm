# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config

logger = init_logger(__name__)


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings used to
    enter and forcibly terminate reasoning. The end string may include a
    transition phrase before the parser's natural reasoning end marker. Token
    IDs are derived automatically by `initialize_token_ids`.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String forced when the thinking budget is exhausted."""
    reasoning_marker_strs: list[str] = field(default_factory=list)
    """Hesitation strings emitted while reasoning, discouraged by
    `reasoning_marker_penalty`. A marker may span multiple tokens, such as
    `"let me think"`."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for forced reasoning end token IDs. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_marker_token_ids: list[list[int]] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_marker_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _natural_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs that naturally terminate reasoning, as defined by the parser."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs forced when the thinking budget is exhausted."""
        return self._reasoning_end_token_ids

    @property
    def reasoning_marker_token_ids(self) -> list[list[int]] | None:
        """Token-ID sequences for each configured reasoning marker.

        A marker may span multiple tokens (for example ``"let me think"``);
        consumers match the sequence against recent history and penalise the
        final token, so a marker is discouraged only where it would complete.
        """
        return self._reasoning_marker_token_ids

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs that indicate the model naturally ended reasoning."""
        return self._natural_reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
            and self._natural_reasoning_end_token_ids is not None
            and self._reasoning_marker_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        if tokenizer is None:
            # ``skip_tokenizer_init`` leaves no tokenizer to encode with, so the
            # token IDs stay uninitialized and reasoning stays disabled.
            return
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_reasoning_end_str = ""
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            natural_reasoning_end_str = end_token or ""

        if not natural_reasoning_end_str:
            natural_reasoning_end_str = reasoning_end_str

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )
        self._natural_reasoning_end_token_ids = tokenizer.encode(
            natural_reasoning_end_str, add_special_tokens=False
        )
        marker_token_ids: list[list[int]] = []
        for marker in self.reasoning_marker_strs:
            if not marker or not marker.strip():
                logger.info("Dropping empty reasoning marker.")
                continue
            token_ids = tokenizer.encode(marker, add_special_tokens=False)
            if not token_ids:
                logger.info(
                    "Dropping reasoning marker %r: encodes to no tokens.", marker
                )
                continue
            if token_ids not in marker_token_ids:
                marker_token_ids.append(token_ids)
        self._reasoning_marker_token_ids = marker_token_ids

        if (
            not self._reasoning_start_token_ids
            or not self._reasoning_end_token_ids
            or not self._natural_reasoning_end_token_ids
        ):
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
