"""
Streaming support for the Parrot Model.

This module implements streaming functionality for both synchronous
and asynchronous response generation, with configurable delays and
chunk sizes to simulate real LLM streaming behavior.
"""


class StreamHelper:
    """
    Utility class for streaming text with configurable delays and chunk sizes.

    StreamHelper provides utilities for breaking text into chunks and managing
    delays between chunks to simulate realistic LLM streaming behavior.

    Attributes:
        delay_ms: Delay in milliseconds between streaming chunks.

    Example:
        >>> helper = StreamHelper(delay_ms=100)
        >>> chunks = helper.chunk_text("Hello world", chunk_size=5)
        >>> print(chunks)
        ['Hello', ' worl', 'd']
        >>> delay = helper.get_delay_seconds()
        >>> print(delay)
        0.1
    """

    def __init__(self, delay_ms: int = 50):
        """
        Initialize the StreamHelper.

        Args:
            delay_ms: Delay in milliseconds between streaming chunks.
                Defaults to 50ms for realistic streaming simulation.

        Example:
            >>> helper = StreamHelper(delay_ms=100)
            >>> helper.delay_ms
            100
        """
        self.delay_ms = delay_ms

    def chunk_text(self, text: str, chunk_size: int = 1) -> list[str]:
        """
        Split text into chunks of specified size.

        Breaks the input text into a list of chunks, each containing
        chunk_size characters. The last chunk may be shorter than
        chunk_size if the text length is not evenly divisible.

        Args:
            text: The text to split into chunks.
            chunk_size: Number of characters per chunk. Defaults to 1
                for character-by-character streaming.

        Returns:
            List of text chunks.

        Example:
            >>> helper = StreamHelper()
            >>> helper.chunk_text("Hello", chunk_size=1)
            ['H', 'e', 'l', 'l', 'o']
            >>> helper.chunk_text("Hello world", chunk_size=5)
            ['Hello', ' worl', 'd']
            >>> helper.chunk_text("", chunk_size=5)
            []
        """
        if not text:
            return []

        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def get_delay_seconds(self) -> float:
        """
        Convert delay_ms to seconds for use with time.sleep() and asyncio.sleep().

        Returns:
            The delay in seconds as a float.

        Example:
            >>> helper = StreamHelper(delay_ms=50)
            >>> helper.get_delay_seconds()
            0.05
            >>> helper = StreamHelper(delay_ms=100)
            >>> helper.get_delay_seconds()
            0.1
        """
        return self.delay_ms / 1000.0
