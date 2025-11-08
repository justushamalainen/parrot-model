"""
Base ParrotModel implementation.

This module contains the core ParrotModel class that implements
the fundamental "parrot" behavior - echoing back messages with
configurable transformations and features.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

from parrot_model.core.config import ParrotConfig
from parrot_model.core.response import ResponseGenerator
from parrot_model.core.streaming import StreamHelper


class ParrotModel:
    """
    The main Parrot Model class for local LLM simulation.

    ParrotModel provides a simple, deterministic interface for simulating LLM
    behavior without requiring API calls or internet connectivity. It "parrots"
    back responses based on configured behavior (currently echo mode).

    This class serves as the foundation for framework-specific adapters and
    can be used directly for simple use cases.

    Attributes:
        config: Configuration object controlling model behavior.
        _generator: Internal ResponseGenerator instance for creating responses.
        tool_calls: List of tool calls from the most recent generate() call.

    Example:
        >>> # Basic usage
        >>> model = ParrotModel()
        >>> response = model.generate("Hello, world!")
        >>> print(response)
        'Hello, world!'

        >>> # With configuration
        >>> config = ParrotConfig(max_tokens=10)
        >>> model = ParrotModel(config=config)
        >>> response = model.generate("This is a very long message that will be truncated")
        >>> print(response)
        'This is a very long message that will'

        >>> # With tool calls
        >>> model = ParrotModel()
        >>> response = model.generate("Get [TOOL:get_weather|city=London]")
        >>> model.get_tool_calls()[0].name
        'get_weather'
    """

    def __init__(self, config: ParrotConfig | None = None):
        """
        Initialize the ParrotModel.

        Args:
            config: Optional configuration object. If None, uses default ParrotConfig.

        Example:
            >>> model = ParrotModel()  # Use defaults
            >>> model = ParrotModel(ParrotConfig(max_chars=100))  # Custom config
        """
        self.config = config or ParrotConfig()
        self._generator = ResponseGenerator(self.config)
        self._stream_helper = StreamHelper(delay_ms=self.config.stream_delay_ms)
        self.tool_calls: list = []  # Stores tool calls from most recent generate()

    def generate(self, message: str) -> str:
        """
        Generate a response synchronously.

        This is the primary method for generating responses. In echo mode,
        it returns the input message, optionally applying text limits based
        on the configuration. If tool calls are enabled, they are parsed and
        stored for later retrieval via get_tool_calls().

        Args:
            message: The input message to generate a response for.

        Returns:
            The generated response string.

        Raises:
            ValueError: If the configured mode is not supported.

        Example:
            >>> model = ParrotModel()
            >>> model.generate("Hello!")
            'Hello!'

            >>> model = ParrotModel(ParrotConfig(max_tokens=2))
            >>> model.generate("Hello beautiful world")
            'Hello beautiful'

            >>> # With tool calls
            >>> model = ParrotModel()
            >>> response = model.generate("Check [TOOL:get_weather|city=London]")
            >>> response
            'Check'
            >>> len(model.get_tool_calls())
            1
        """
        # Parse tool calls once if enabled, getting clean message
        if self.config.enable_tool_calls:
            clean_message, self.tool_calls = self._generator.parse_tool_calls(message)
        else:
            clean_message = message
            self.tool_calls = []

        # Generate the response from clean message
        return self._generator.generate(clean_message)

    async def agenerate(self, message: str) -> str:
        """
        Generate a response asynchronously.

        This is the async version of generate(). Currently, it's a simple
        async wrapper around the synchronous generation logic. In future
        implementations, this may support true async streaming or processing.

        Args:
            message: The input message to generate a response for.

        Returns:
            The generated response string.

        Raises:
            ValueError: If the configured mode is not supported.

        Example:
            >>> import asyncio
            >>> model = ParrotModel()
            >>> response = asyncio.run(model.agenerate("Hello async world!"))
            >>> print(response)
            'Hello async world!'
        """
        # For now, just call the synchronous version
        # In the future, this could support async streaming or processing
        return self.generate(message)

    def stream(self, message: str, chunk_size: int = 1) -> Iterator[str]:
        """
        Generate a streaming response synchronously.

        This method simulates LLM streaming by generating the complete response
        first, then yielding it in chunks with configurable delays between each
        chunk. This provides a realistic streaming experience for development
        and testing.

        Args:
            message: The input message to generate a response for.
            chunk_size: Number of characters per chunk. Defaults to 1 for
                character-by-character streaming. Can be increased for faster
                streaming or word-by-word simulation.

        Yields:
            String chunks of the response.

        Raises:
            ValueError: If the configured mode is not supported.

        Example:
            >>> model = ParrotModel()
            >>> for chunk in model.stream("Hello"):
            ...     print(chunk, end="", flush=True)
            Hello

            >>> # With custom chunk size
            >>> model = ParrotModel()
            >>> chunks = list(model.stream("Hello world", chunk_size=5))
            >>> print(chunks)
            ['Hello', ' worl', 'd']
        """
        # Generate the full response first
        response = self.generate(message)

        # Split into chunks and yield with delays
        chunks = self._stream_helper.chunk_text(response, chunk_size=chunk_size)
        delay = self._stream_helper.get_delay_seconds()

        for i, chunk in enumerate(chunks):
            yield chunk
            # Don't sleep after the last chunk
            if i < len(chunks) - 1:
                time.sleep(delay)

    async def astream(self, message: str, chunk_size: int = 1) -> AsyncIterator[str]:
        """
        Generate a streaming response asynchronously.

        This is the async version of stream(). It generates the complete response
        first using agenerate(), then yields it in chunks with async delays between
        each chunk. This provides a realistic async streaming experience for
        development and testing.

        Args:
            message: The input message to generate a response for.
            chunk_size: Number of characters per chunk. Defaults to 1 for
                character-by-character streaming. Can be increased for faster
                streaming or word-by-word simulation.

        Yields:
            String chunks of the response.

        Raises:
            ValueError: If the configured mode is not supported.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     model = ParrotModel()
            ...     async for chunk in model.astream("Hello"):
            ...         print(chunk, end="", flush=True)
            >>> asyncio.run(main())
            Hello

            >>> # With custom chunk size
            >>> async def main():
            ...     model = ParrotModel()
            ...     chunks = []
            ...     async for chunk in model.astream("Hello world", chunk_size=5):
            ...         chunks.append(chunk)
            ...     print(chunks)
            >>> asyncio.run(main())
            ['Hello', ' worl', 'd']
        """
        # Generate the full response first (async)
        response = await self.agenerate(message)

        # Split into chunks and yield with async delays
        chunks = self._stream_helper.chunk_text(response, chunk_size=chunk_size)
        delay = self._stream_helper.get_delay_seconds()

        for i, chunk in enumerate(chunks):
            yield chunk
            # Don't sleep after the last chunk
            if i < len(chunks) - 1:
                await asyncio.sleep(delay)

    def get_tool_calls(self) -> list:
        """
        Get the tool calls from the most recent generate() call.

        Returns the list of ToolCall objects that were parsed from the most
        recent call to generate(), agenerate(), stream(), or astream(). If no
        tool calls were found or if tool calls are disabled, returns an empty list.

        Returns:
            List of ToolCall objects from the most recent generation.

        Example:
            >>> model = ParrotModel()
            >>> response = model.generate("Get [TOOL:get_weather|city=London]")
            >>> tool_calls = model.get_tool_calls()
            >>> len(tool_calls)
            1
            >>> tool_calls[0].name
            'get_weather'
            >>> tool_calls[0].parameters
            {'city': 'London'}

            >>> # Multiple tool calls
            >>> response = model.generate("[TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris]")
            >>> len(model.get_tool_calls())
            2
        """
        return self.tool_calls
