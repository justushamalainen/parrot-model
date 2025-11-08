"""
OpenAI SDK adapter for the Parrot Model.

This module implements the OpenAI Python SDK interface, allowing the Parrot Model
to be used as a drop-in replacement for the OpenAI client. It mimics the OpenAI
chat completions API, including streaming and async support.

Example:
    >>> from parrot_model.adapters.openai import OpenAI
    >>>
    >>> # Use just like the real OpenAI client
    >>> client = OpenAI()
    >>> response = client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[
    ...         {"role": "system", "content": "You are helpful."},
    ...         {"role": "user", "content": "Hello!"}
    ...     ]
    ... )
    >>> print(response.choices[0].message.content)
    'Hello!'
    >>>
    >>> # With streaming
    >>> stream = client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     stream=True
    ... )
    >>> for chunk in stream:
    ...     if chunk.choices[0].delta.content:
    ...         print(chunk.choices[0].delta.content, end="")
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


class ChatCompletionMessage:
    """Mimics OpenAI's ChatCompletionMessage."""

    def __init__(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

    def __repr__(self) -> str:
        return f"ChatCompletionMessage(role={self.role!r}, content={self.content!r})"


class Choice:
    """Mimics OpenAI's Choice object."""

    def __init__(
        self,
        index: int,
        message: ChatCompletionMessage,
        finish_reason: str = "stop",
    ):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return f"Choice(index={self.index}, message={self.message}, finish_reason={self.finish_reason!r})"


class ChatCompletion:
    """Mimics OpenAI's ChatCompletion response."""

    def __init__(
        self,
        id: str,
        choices: list[Choice],
        created: int,
        model: str,
        object: str = "chat.completion",
    ):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.object = object

    def __repr__(self) -> str:
        return f"ChatCompletion(id={self.id!r}, choices={self.choices})"


class ChatCompletionChunkDelta:
    """Mimics OpenAI's streaming delta object."""

    def __init__(
        self,
        role: str | None = None,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

    def __repr__(self) -> str:
        return f"ChatCompletionChunkDelta(role={self.role!r}, content={self.content!r})"


class ChatCompletionChunkChoice:
    """Mimics OpenAI's streaming choice object."""

    def __init__(
        self,
        index: int,
        delta: ChatCompletionChunkDelta,
        finish_reason: str | None = None,
    ):
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return f"ChatCompletionChunkChoice(index={self.index}, delta={self.delta})"


class ChatCompletionChunk:
    """Mimics OpenAI's streaming chunk response."""

    def __init__(
        self,
        id: str,
        choices: list[ChatCompletionChunkChoice],
        created: int,
        model: str,
        object: str = "chat.completion.chunk",
    ):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.object = object

    def __repr__(self) -> str:
        return f"ChatCompletionChunk(id={self.id!r})"


class Completions:
    """Mimics OpenAI's chat.completions interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self._parrot_config = parrot_config or ParrotConfig()
        self._parrot_model = ParrotModel(config=self._parrot_config)

    def _convert_messages_to_string(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Convert OpenAI message format to a string for ParrotModel."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip system messages unless configured to echo them
            if role == "system" and not self._parrot_config.echo_system_messages:
                continue

            parts.append(str(content))

        return " ".join(parts)

    def _format_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Format ParrotModel tool calls to OpenAI format."""
        formatted_calls = []
        for tc in tool_calls:
            formatted_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.parameters),
                    },
                }
            )
        return formatted_calls

    def _generate_completion_id(self, prompt: str, model: str) -> str:
        """Generate a completion ID based on deterministic config."""
        if self._parrot_config.deterministic:
            return generate_deterministic_id(
                prompt, model, prefix="chatcmpl-", length=8
            )
        else:
            return f"chatcmpl-{uuid.uuid4().hex[:8]}"

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
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """
        Create a chat completion.

        Args:
            model: Model name (ignored, for API compatibility).
            messages: List of message dicts with 'role' and 'content'.
            stream: Whether to stream the response.
            temperature: Temperature setting (ignored, for API compatibility).
            max_tokens: Maximum tokens (used if configured).
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            ChatCompletion object or streaming iterator.

        Example:
            >>> client = OpenAI()
            >>> response = client.chat.completions.create(
            ...     model="gpt-4",
            ...     messages=[{"role": "user", "content": "Hello!"}]
            ... )
            >>> print(response.choices[0].message.content)
            'Hello!'
        """
        # Convert messages to string
        prompt = self._convert_messages_to_string(messages)

        # Handle streaming
        if stream:
            return self._create_stream(prompt, model)

        # Generate response
        response_text = self._parrot_model.generate(prompt)
        tool_calls = self._parrot_model.get_tool_calls()

        # Create response message
        message = ChatCompletionMessage(
            role="assistant",
            content=response_text,
            tool_calls=self._format_tool_calls(tool_calls) if tool_calls else None,
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

        # Create completion
        completion = ChatCompletion(
            id=self._generate_completion_id(prompt, model),
            choices=[choice],
            created=self._get_timestamp(),
            model=model,
        )

        return completion

    def _create_stream(
        self, prompt: str, model: str
    ) -> Iterator[ChatCompletionChunk]:
        """Create a streaming response."""
        completion_id = self._generate_completion_id(prompt, model)
        created = self._get_timestamp()

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
            created=created,
            model=model,
        )

        # Stream content chunks
        for chunk in self._parrot_model.stream(prompt):
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=chunk),
                        finish_reason=None,
                    )
                ],
                created=created,
                model=model,
            )

        # Final chunk with finish_reason
        tool_calls = self._parrot_model.get_tool_calls()
        if tool_calls:
            # Include tool calls in final chunk
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(
                            tool_calls=self._format_tool_calls(tool_calls)
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                created=created,
                model=model,
            )
        else:
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
                created=created,
                model=model,
            )


class AsyncCompletions:
    """Mimics OpenAI's async chat.completions interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self._parrot_config = parrot_config or ParrotConfig()
        self._parrot_model = ParrotModel(config=self._parrot_config)

    def _convert_messages_to_string(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Convert OpenAI message format to a string for ParrotModel."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip system messages unless configured to echo them
            if role == "system" and not self._parrot_config.echo_system_messages:
                continue

            parts.append(str(content))

        return " ".join(parts)

    def _format_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Format ParrotModel tool calls to OpenAI format."""
        formatted_calls = []
        for tc in tool_calls:
            formatted_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.parameters),
                    },
                }
            )
        return formatted_calls

    def _generate_completion_id(self, prompt: str, model: str) -> str:
        """Generate a completion ID based on deterministic config."""
        if self._parrot_config.deterministic:
            return generate_deterministic_id(
                prompt, model, prefix="chatcmpl-", length=8
            )
        else:
            return f"chatcmpl-{uuid.uuid4().hex[:8]}"

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
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """
        Create a chat completion asynchronously.

        Args:
            model: Model name (ignored, for API compatibility).
            messages: List of message dicts with 'role' and 'content'.
            stream: Whether to stream the response.
            temperature: Temperature setting (ignored, for API compatibility).
            max_tokens: Maximum tokens (used if configured).
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            ChatCompletion object or async streaming iterator.

        Example:
            >>> import asyncio
            >>> client = AsyncOpenAI()
            >>> async def main():
            ...     response = await client.chat.completions.create(
            ...         model="gpt-4",
            ...         messages=[{"role": "user", "content": "Hello!"}]
            ...     )
            ...     print(response.choices[0].message.content)
            >>> asyncio.run(main())
            'Hello!'
        """
        # Convert messages to string
        prompt = self._convert_messages_to_string(messages)

        # Handle streaming
        if stream:
            return self._create_stream(prompt, model)

        # Generate response
        response_text = await self._parrot_model.agenerate(prompt)
        tool_calls = self._parrot_model.get_tool_calls()

        # Create response message
        message = ChatCompletionMessage(
            role="assistant",
            content=response_text,
            tool_calls=self._format_tool_calls(tool_calls) if tool_calls else None,
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

        # Create completion
        completion = ChatCompletion(
            id=self._generate_completion_id(prompt, model),
            choices=[choice],
            created=self._get_timestamp(),
            model=model,
        )

        return completion

    async def _create_stream(
        self, prompt: str, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create an async streaming response."""
        completion_id = self._generate_completion_id(prompt, model)
        created = self._get_timestamp()

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
            created=created,
            model=model,
        )

        # Stream content chunks
        async for chunk in self._parrot_model.astream(prompt):
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=chunk),
                        finish_reason=None,
                    )
                ],
                created=created,
                model=model,
            )

        # Final chunk with finish_reason
        tool_calls = self._parrot_model.get_tool_calls()
        if tool_calls:
            # Include tool calls in final chunk
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(
                            tool_calls=self._format_tool_calls(tool_calls)
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                created=created,
                model=model,
            )
        else:
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
                created=created,
                model=model,
            )


class Chat:
    """Mimics OpenAI's chat interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self.completions = Completions(parrot_config)


class AsyncChat:
    """Mimics OpenAI's async chat interface."""

    def __init__(self, parrot_config: ParrotConfig | None = None):
        self.completions = AsyncCompletions(parrot_config)


class OpenAI:
    """
    Mimics the OpenAI client for use with ParrotModel.

    This class provides a drop-in replacement for the OpenAI Python SDK,
    allowing you to use ParrotModel with the same interface as the real
    OpenAI client.

    Attributes:
        chat: Chat interface with completions endpoint.

    Example:
        >>> from parrot_model.adapters.openai import OpenAI
        >>>
        >>> # Use just like the real OpenAI client
        >>> client = OpenAI()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[
        ...         {"role": "system", "content": "You are helpful."},
        ...         {"role": "user", "content": "Hello!"}
        ...     ]
        ... )
        >>> print(response.choices[0].message.content)
        'Hello!'
        >>>
        >>> # With custom configuration
        >>> from parrot_model import ParrotConfig
        >>> client = OpenAI(parrot_config=ParrotConfig(max_tokens=50))
        >>>
        >>> # With streaming
        >>> stream = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     stream=True
        ... )
        >>> for chunk in stream:
        ...     if chunk.choices[0].delta.content:
        ...         print(chunk.choices[0].delta.content, end="")
    """

    def __init__(
        self,
        api_key: str | None = None,
        parrot_config: ParrotConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI client mock.

        Args:
            api_key: API key (ignored, for compatibility).
            parrot_config: Optional ParrotConfig for customizing behavior.
            **kwargs: Additional arguments (ignored, for compatibility).

        Example:
            >>> client = OpenAI()
            >>> client = OpenAI(parrot_config=ParrotConfig(max_tokens=100))
        """
        self._parrot_config = parrot_config or ParrotConfig()
        self.chat = Chat(self._parrot_config)


class AsyncOpenAI:
    """
    Mimics the async OpenAI client for use with ParrotModel.

    This class provides a drop-in replacement for the async OpenAI Python SDK,
    allowing you to use ParrotModel with the same interface as the real
    async OpenAI client.

    Attributes:
        chat: Async chat interface with completions endpoint.

    Example:
        >>> import asyncio
        >>> from parrot_model.adapters.openai import AsyncOpenAI
        >>>
        >>> async def main():
        ...     client = AsyncOpenAI()
        ...     response = await client.chat.completions.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     print(response.choices[0].message.content)
        >>>
        >>> asyncio.run(main())
        'Hello!'
    """

    def __init__(
        self,
        api_key: str | None = None,
        parrot_config: ParrotConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the async OpenAI client mock.

        Args:
            api_key: API key (ignored, for compatibility).
            parrot_config: Optional ParrotConfig for customizing behavior.
            **kwargs: Additional arguments (ignored, for compatibility).

        Example:
            >>> client = AsyncOpenAI()
            >>> client = AsyncOpenAI(parrot_config=ParrotConfig(max_tokens=100))
        """
        self._parrot_config = parrot_config or ParrotConfig()
        self.chat = AsyncChat(self._parrot_config)


# Export main classes
__all__ = [
    "OpenAI",
    "AsyncOpenAI",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
]
