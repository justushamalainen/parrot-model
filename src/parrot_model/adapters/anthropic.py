"""
Anthropic SDK adapter for the Parrot Model.

This module implements the Anthropic Python SDK interface, allowing the Parrot Model
to be used as a drop-in replacement for the Anthropic client. It mimics the Anthropic
messages API, including streaming and async support.

Example:
    >>> from parrot_model.adapters.anthropic import Anthropic
    >>>
    >>> # Use just like the real Anthropic client
    >>> client = Anthropic()
    >>> message = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     max_tokens=1024,
    ...     messages=[
    ...         {"role": "user", "content": "Hello, Claude"}
    ...     ]
    ... )
    >>> print(message.content[0].text)
    'Hello, Claude'
    >>>
    >>> # With streaming
    >>> stream = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     max_tokens=1024,
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     stream=True
    ... )
    >>> for event in stream:
    ...     if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
    ...         print(event.delta.text, end="")
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig
from parrot_model.utils.deterministic import (
    generate_deterministic_id,
    get_deterministic_timestamp,
)


class TextBlock:
    """Mimics Anthropic's TextBlock."""

    def __init__(self, type: str = "text", text: str = ""):
        self.type = type
        self.text = text

    def __repr__(self) -> str:
        return f"TextBlock(type={self.type!r}, text={self.text!r})"


class ToolUseBlock:
    """Mimics Anthropic's ToolUseBlock."""

    def __init__(
        self,
        id: str,
        type: str = "tool_use",
        name: str = "",
        input: dict[str, Any] | None = None,
    ):
        self.id = id
        self.type = type
        self.name = name
        self.input = input or {}

    def __repr__(self) -> str:
        return f"ToolUseBlock(id={self.id!r}, name={self.name!r})"


class Usage:
    """Mimics Anthropic's Usage."""

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __repr__(self) -> str:
        return f"Usage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens})"


class Message:
    """Mimics Anthropic's Message response."""

    def __init__(
        self,
        id: str,
        type: str,
        role: str,
        content: list[TextBlock | ToolUseBlock],
        model: str,
        stop_reason: str | None,
        stop_sequence: str | None = None,
        usage: Usage | None = None,
    ):
        self.id = id
        self.type = type
        self.role = role
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.usage = usage or Usage()

    def __repr__(self) -> str:
        return f"Message(id={self.id!r}, role={self.role!r})"


# Streaming event types
class MessageStartEvent:
    """Mimics Anthropic's MessageStartEvent."""

    def __init__(self, message: Message):
        self.type = "message_start"
        self.message = message


class ContentBlockStartEvent:
    """Mimics Anthropic's ContentBlockStartEvent."""

    def __init__(self, index: int, content_block: TextBlock | ToolUseBlock):
        self.type = "content_block_start"
        self.index = index
        self.content_block = content_block


class ContentBlockDelta:
    """Mimics Anthropic's content block delta."""

    def __init__(self, type: str = "text_delta", text: str = ""):
        self.type = type
        self.text = text


class ContentBlockDeltaEvent:
    """Mimics Anthropic's ContentBlockDeltaEvent."""

    def __init__(self, index: int, delta: ContentBlockDelta):
        self.type = "content_block_delta"
        self.index = index
        self.delta = delta


class ContentBlockStopEvent:
    """Mimics Anthropic's ContentBlockStopEvent."""

    def __init__(self, index: int):
        self.type = "content_block_stop"
        self.index = index


class MessageDelta:
    """Mimics Anthropic's message delta."""

    def __init__(self, stop_reason: str | None = None, stop_sequence: str | None = None):
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence


class MessageDeltaEvent:
    """Mimics Anthropic's MessageDeltaEvent."""

    def __init__(self, delta: MessageDelta, usage: Usage):
        self.type = "message_delta"
        self.delta = delta
        self.usage = usage


class MessageStopEvent:
    """Mimics Anthropic's MessageStopEvent."""

    def __init__(self):
        self.type = "message_stop"


class Messages:
    """Mimics Anthropic's messages interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self._parrot_config = parrot_config or ParrotConfig()
        self._parrot_model = ParrotModel(config=self._parrot_config)

    def _convert_messages_to_string(self, messages: list[dict[str, Any]]) -> str:
        """Convert Anthropic message format to a string for ParrotModel."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle both string and list content
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                content = " ".join(text_parts)

            parts.append(str(content))

        return " ".join(parts)

    def _format_tool_use_blocks(self, tool_calls: list[Any]) -> list[ToolUseBlock]:
        """Format ParrotModel tool calls to Anthropic ToolUseBlock format."""
        blocks = []
        for tc in tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=tc.id,
                    name=tc.name,
                    input=tc.parameters,
                )
            )
        return blocks

    def _generate_message_id(self, prompt: str, model: str) -> str:
        """Generate a message ID based on deterministic config."""
        if self._parrot_config.deterministic:
            return generate_deterministic_id(
                prompt, model, prefix="msg_", length=24
            )
        else:
            return f"msg_{uuid.uuid4().hex[:24]}"

    def _get_timestamp(self) -> int:
        """Get timestamp based on deterministic config."""
        if self._parrot_config.deterministic:
            return get_deterministic_timestamp()
        else:
            return int(time.time())

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Message | Iterator[Any]:
        """
        Create a message.

        Args:
            model: Model name (ignored, for API compatibility).
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens (used if configured).
            system: System prompt (ignored unless configured to echo).
            stream: Whether to stream the response.
            temperature: Temperature setting (ignored, for API compatibility).
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            Message object or streaming iterator.

        Example:
            >>> client = Anthropic()
            >>> message = client.messages.create(
            ...     model="claude-3-5-sonnet-20241022",
            ...     max_tokens=1024,
            ...     messages=[{"role": "user", "content": "Hello!"}]
            ... )
            >>> print(message.content[0].text)
            'Hello!'
        """
        # Convert messages to string
        prompt = self._convert_messages_to_string(messages)

        # Add system message if provided and configured
        if system and self._parrot_config.echo_system_messages:
            prompt = f"System: {system}\n{prompt}"

        # Handle streaming
        if stream:
            return self._create_stream(prompt, model)

        # Generate response
        response_text = self._parrot_model.generate(prompt)
        tool_calls = self._parrot_model.get_tool_calls()

        # Build content blocks
        content: list[TextBlock | ToolUseBlock] = []

        # Add text block if there's content
        if response_text:
            content.append(TextBlock(text=response_text))

        # Add tool use blocks if present
        if tool_calls:
            content.extend(self._format_tool_use_blocks(tool_calls))

        # Determine stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Create message
        message = Message(
            id=self._generate_message_id(prompt, model),
            type="message",
            role="assistant",
            content=content,
            model=model,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=10, output_tokens=20),  # Mock values
        )

        return message

    def _create_stream(self, prompt: str, model: str) -> Iterator[Any]:
        """Create a streaming response."""
        message_id = self._generate_message_id(prompt, model)

        # Message start event
        yield MessageStartEvent(
            message=Message(
                id=message_id,
                type="message",
                role="assistant",
                content=[],
                model=model,
                stop_reason=None,
                usage=Usage(input_tokens=10, output_tokens=0),
            )
        )

        # Content block start
        yield ContentBlockStartEvent(index=0, content_block=TextBlock())

        # Stream text deltas
        for chunk in self._parrot_model.stream(prompt):
            yield ContentBlockDeltaEvent(
                index=0, delta=ContentBlockDelta(text=chunk)
            )

        # Content block stop
        yield ContentBlockStopEvent(index=0)

        # Check for tool calls
        tool_calls = self._parrot_model.get_tool_calls()
        if tool_calls:
            # Add tool use blocks
            for i, tc in enumerate(tool_calls, start=1):
                tool_block = ToolUseBlock(id=tc.id, name=tc.name, input=tc.parameters)
                yield ContentBlockStartEvent(index=i, content_block=tool_block)
                yield ContentBlockStopEvent(index=i)

        # Message delta with stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"
        yield MessageDeltaEvent(
            delta=MessageDelta(stop_reason=stop_reason),
            usage=Usage(output_tokens=20),
        )

        # Message stop
        yield MessageStopEvent()


class AsyncMessages:
    """Mimics Anthropic's async messages interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self._parrot_config = parrot_config or ParrotConfig()
        self._parrot_model = ParrotModel(config=self._parrot_config)

    def _convert_messages_to_string(self, messages: list[dict[str, Any]]) -> str:
        """Convert Anthropic message format to a string for ParrotModel."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle both string and list content
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                content = " ".join(text_parts)

            parts.append(str(content))

        return " ".join(parts)

    def _format_tool_use_blocks(self, tool_calls: list[Any]) -> list[ToolUseBlock]:
        """Format ParrotModel tool calls to Anthropic ToolUseBlock format."""
        blocks = []
        for tc in tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=tc.id,
                    name=tc.name,
                    input=tc.parameters,
                )
            )
        return blocks

    def _generate_message_id(self, prompt: str, model: str) -> str:
        """Generate a message ID based on deterministic config."""
        if self._parrot_config.deterministic:
            return generate_deterministic_id(
                prompt, model, prefix="msg_", length=24
            )
        else:
            return f"msg_{uuid.uuid4().hex[:24]}"

    def _get_timestamp(self) -> int:
        """Get timestamp based on deterministic config."""
        if self._parrot_config.deterministic:
            return get_deterministic_timestamp()
        else:
            return int(time.time())

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Message | AsyncIterator[Any]:
        """
        Create a message asynchronously.

        Args:
            model: Model name (ignored, for API compatibility).
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens (used if configured).
            system: System prompt (ignored unless configured to echo).
            stream: Whether to stream the response.
            temperature: Temperature setting (ignored, for API compatibility).
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            Message object or async streaming iterator.

        Example:
            >>> import asyncio
            >>> client = AsyncAnthropic()
            >>> async def main():
            ...     message = await client.messages.create(
            ...         model="claude-3-5-sonnet-20241022",
            ...         max_tokens=1024,
            ...         messages=[{"role": "user", "content": "Hello!"}]
            ...     )
            ...     print(message.content[0].text)
            >>> asyncio.run(main())
            'Hello!'
        """
        # Convert messages to string
        prompt = self._convert_messages_to_string(messages)

        # Add system message if provided and configured
        if system and self._parrot_config.echo_system_messages:
            prompt = f"System: {system}\n{prompt}"

        # Handle streaming
        if stream:
            return self._create_stream(prompt, model)

        # Generate response
        response_text = await self._parrot_model.agenerate(prompt)
        tool_calls = self._parrot_model.get_tool_calls()

        # Build content blocks
        content: list[TextBlock | ToolUseBlock] = []

        # Add text block if there's content
        if response_text:
            content.append(TextBlock(text=response_text))

        # Add tool use blocks if present
        if tool_calls:
            content.extend(self._format_tool_use_blocks(tool_calls))

        # Determine stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Create message
        message = Message(
            id=self._generate_message_id(prompt, model),
            type="message",
            role="assistant",
            content=content,
            model=model,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=10, output_tokens=20),  # Mock values
        )

        return message

    async def _create_stream(self, prompt: str, model: str) -> AsyncIterator[Any]:
        """Create an async streaming response."""
        message_id = self._generate_message_id(prompt, model)

        # Message start event
        yield MessageStartEvent(
            message=Message(
                id=message_id,
                type="message",
                role="assistant",
                content=[],
                model=model,
                stop_reason=None,
                usage=Usage(input_tokens=10, output_tokens=0),
            )
        )

        # Content block start
        yield ContentBlockStartEvent(index=0, content_block=TextBlock())

        # Stream text deltas
        async for chunk in self._parrot_model.astream(prompt):
            yield ContentBlockDeltaEvent(
                index=0, delta=ContentBlockDelta(text=chunk)
            )

        # Content block stop
        yield ContentBlockStopEvent(index=0)

        # Check for tool calls
        tool_calls = self._parrot_model.get_tool_calls()
        if tool_calls:
            # Add tool use blocks
            for i, tc in enumerate(tool_calls, start=1):
                tool_block = ToolUseBlock(id=tc.id, name=tc.name, input=tc.parameters)
                yield ContentBlockStartEvent(index=i, content_block=tool_block)
                yield ContentBlockStopEvent(index=i)

        # Message delta with stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"
        yield MessageDeltaEvent(
            delta=MessageDelta(stop_reason=stop_reason),
            usage=Usage(output_tokens=20),
        )

        # Message stop
        yield MessageStopEvent()


class Anthropic:
    """
    Mimics the Anthropic client for use with ParrotModel.

    This class provides a drop-in replacement for the Anthropic Python SDK,
    allowing you to use ParrotModel with the same interface as the real
    Anthropic client.

    Attributes:
        messages: Messages interface for creating completions.

    Example:
        >>> from parrot_model.adapters.anthropic import Anthropic
        >>>
        >>> # Use just like the real Anthropic client
        >>> client = Anthropic()
        >>> message = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[
        ...         {"role": "user", "content": "Hello, Claude"}
        ...     ]
        ... )
        >>> print(message.content[0].text)
        'Hello, Claude'
        >>>
        >>> # With custom configuration
        >>> from parrot_model import ParrotConfig
        >>> client = Anthropic(parrot_config=ParrotConfig(max_tokens=50))
        >>>
        >>> # With streaming
        >>> stream = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     stream=True
        ... )
        >>> for event in stream:
        ...     if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
        ...         print(event.delta.text, end="")
    """

    def __init__(
        self,
        api_key: str | None = None,
        parrot_config: ParrotConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic client mock.

        Args:
            api_key: API key (ignored, for compatibility).
            parrot_config: Optional ParrotConfig for customizing behavior.
            **kwargs: Additional arguments (ignored, for compatibility).

        Example:
            >>> client = Anthropic()
            >>> client = Anthropic(parrot_config=ParrotConfig(max_tokens=100))
        """
        self._parrot_config = parrot_config or ParrotConfig()
        self.messages = Messages(self._parrot_config)


class AsyncAnthropic:
    """
    Mimics the async Anthropic client for use with ParrotModel.

    This class provides a drop-in replacement for the async Anthropic Python SDK,
    allowing you to use ParrotModel with the same interface as the real
    async Anthropic client.

    Attributes:
        messages: Async messages interface for creating completions.

    Example:
        >>> import asyncio
        >>> from parrot_model.adapters.anthropic import AsyncAnthropic
        >>>
        >>> async def main():
        ...     client = AsyncAnthropic()
        ...     message = await client.messages.create(
        ...         model="claude-3-5-sonnet-20241022",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hello, Claude"}]
        ...     )
        ...     print(message.content[0].text)
        >>>
        >>> asyncio.run(main())
        'Hello, Claude'
    """

    def __init__(
        self,
        api_key: str | None = None,
        parrot_config: ParrotConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the async Anthropic client mock.

        Args:
            api_key: API key (ignored, for compatibility).
            parrot_config: Optional ParrotConfig for customizing behavior.
            **kwargs: Additional arguments (ignored, for compatibility).

        Example:
            >>> client = AsyncAnthropic()
            >>> client = AsyncAnthropic(parrot_config=ParrotConfig(max_tokens=100))
        """
        self._parrot_config = parrot_config or ParrotConfig()
        self.messages = AsyncMessages(self._parrot_config)


# Export main classes
__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "Message",
    "TextBlock",
    "ToolUseBlock",
]
