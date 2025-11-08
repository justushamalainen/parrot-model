"""
Configuration classes for the Parrot Model.

This module defines the ParrotConfig class and related configuration
options for controlling the behavior of the Parrot Model, including
response modes, text limiting, streaming delays, and tool call handling.
"""

from dataclasses import dataclass


@dataclass
class ParrotConfig:
    """
    Configuration options for the Parrot Model.

    This dataclass defines all configurable parameters that control the behavior
    of the Parrot Model, including response generation mode, text limiting,
    streaming settings, and tool call handling.

    Attributes:
        mode: Response generation mode. Currently supports:
            - "echo": Returns the input message as-is (default)
            - Future: "template", "smart"
        max_tokens: Maximum number of tokens in the response. If set, responses
            will be truncated to this length using simple whitespace-based tokenization.
            None means no token limit.
        max_chars: Maximum number of characters in the response. If set, responses
            will be truncated to this length. None means no character limit.
        truncate_at_word: Whether to truncate at word boundaries when applying
            text limits. If True, the truncation will occur at the last complete
            word before the limit. If False, truncation occurs exactly at the limit.
        stream_delay_ms: Delay in milliseconds between streaming chunks when
            using streaming mode. Simulates realistic LLM streaming behavior.
        enable_tool_calls: Whether to parse and handle tool call encodings from
            messages. When True, messages containing the tool call pattern will
            be parsed to extract tool calls.
        tool_call_pattern: Regular expression pattern used to detect and parse
            tool calls from messages. Default pattern matches: [TOOL:name|params]
        echo_system_messages: Whether to include system messages in echo mode.
            If False (default), system messages are filtered out and only user
            messages are echoed back.
        deterministic: Whether to use deterministic IDs and timestamps. When True
            (default), the same inputs will always produce the same IDs and timestamps,
            making responses fully reproducible. When False, random UUIDs and current
            timestamps are used (useful for more realistic API simulation).

    Example:
        >>> config = ParrotConfig(
        ...     mode="echo",
        ...     max_tokens=100,
        ...     truncate_at_word=True
        ... )
        >>> model = ParrotModel(config=config)
    """

    mode: str = "echo"
    max_tokens: int | None = None
    max_chars: int | None = None
    truncate_at_word: bool = True
    stream_delay_ms: int = 50
    enable_tool_calls: bool = True
    tool_call_pattern: str = r"\[TOOL:(\w+)\|(.+?)\]"
    echo_system_messages: bool = False
    deterministic: bool = True  # Use deterministic IDs and timestamps by default
