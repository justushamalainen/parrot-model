"""
Response generation logic for the Parrot Model.

This module handles the generation of responses based on the configured
mode (echo, template, etc.) and applies text limiting and other transformations.
"""

from parrot_model.core.config import ParrotConfig
from parrot_model.utils.tokenizer import truncate_to_chars, truncate_to_tokens


class ResponseGenerator:
    """
    Generates responses based on configuration settings.

    This class handles the core logic of generating responses from input messages.
    It supports different modes (currently only echo) and applies text limiting
    based on the configured constraints. When tool calls are enabled, it also
    parses and extracts tool call encodings from messages.

    Attributes:
        config: The ParrotConfig instance controlling response generation behavior.
        tool_parser: Optional ToolCallParser for parsing tool call encodings.
                    Created if config.enable_tool_calls is True.

    Example:
        >>> config = ParrotConfig(max_tokens=5)
        >>> generator = ResponseGenerator(config)
        >>> generator.generate("Hello world how are you today")
        'Hello world how are you'
    """

    def __init__(self, config: ParrotConfig):
        """
        Initialize the ResponseGenerator.

        Args:
            config: Configuration object controlling response generation.
        """
        self.config = config
        self.tool_parser = None

        # Create tool parser if tool calls are enabled
        if config.enable_tool_calls:
            # Import here to avoid circular dependency
            from parrot_model.core.tools import ToolCallParser
            self.tool_parser = ToolCallParser(config.tool_call_pattern)

    def parse_tool_calls(self, message: str) -> tuple[str, list]:
        """
        Parse and extract tool calls from a message.

        If tool calls are enabled, extracts all tool call encodings from the
        message and returns both the cleaned message (with tool syntax removed)
        and the list of parsed tool calls.

        Args:
            message: The message potentially containing tool call encodings.

        Returns:
            A tuple of (clean_message, tool_calls) where:
                - clean_message: The message with tool call syntax removed
                - tool_calls: List of ToolCall objects found in the message

        Example:
            >>> config = ParrotConfig(enable_tool_calls=True)
            >>> generator = ResponseGenerator(config)
            >>> msg = "Check [TOOL:get_weather|city=London] today"
            >>> clean_msg, calls = generator.parse_tool_calls(msg)
            >>> clean_msg
            'Check today'
            >>> len(calls)
            1
            >>> calls[0].name
            'get_weather'
        """
        if not self.tool_parser:
            # Tool calls not enabled, return message as-is with empty list
            return message, []

        # Parse tool calls from message
        tool_calls = self.tool_parser.parse(message)

        # Remove tool call syntax from message
        clean_message = self.tool_parser.remove_from_message(message)

        return clean_message, tool_calls

    def generate(self, message: str) -> str:
        """
        Generate a response based on the input message and configuration.

        Currently supports only "echo" mode, which returns the message as-is
        (with optional text limiting applied). If tool calls are enabled,
        they are parsed and the tool call syntax is removed from the response.

        Args:
            message: The input message to generate a response for.

        Returns:
            The generated response string.

        Raises:
            ValueError: If an unsupported mode is configured.

        Example:
            >>> config = ParrotConfig(mode="echo", max_chars=10)
            >>> generator = ResponseGenerator(config)
            >>> generator.generate("Hello world!")
            'Hello'
        """
        # Parse tool calls if enabled (this also cleans the message)
        if self.tool_parser:
            message, _ = self.parse_tool_calls(message)

        # Generate base response based on mode
        if self.config.mode == "echo":
            response = message
        else:
            raise ValueError(
                f"Unsupported mode: {self.config.mode}. "
                f"Currently only 'echo' mode is supported."
            )

        # Apply text limiting if configured
        response = self._apply_text_limits(response)

        return response

    def _apply_text_limits(self, text: str) -> str:
        """
        Apply text limiting constraints to the response.

        Applies token-based and/or character-based limits according to the
        configuration. If both limits are set, token limit is applied first,
        then character limit.

        Args:
            text: The text to apply limits to.

        Returns:
            The text after applying configured limits.

        Example:
            >>> config = ParrotConfig(max_tokens=3, max_chars=10)
            >>> generator = ResponseGenerator(config)
            >>> generator._apply_text_limits("The quick brown fox jumps")
            'The quick'
        """
        if not text:
            return text

        # Apply token limit first if configured
        if self.config.max_tokens is not None:
            text = truncate_to_tokens(
                text, self.config.max_tokens, at_word=self.config.truncate_at_word
            )

        # Apply character limit if configured
        if self.config.max_chars is not None:
            text = truncate_to_chars(
                text, self.config.max_chars, at_word=self.config.truncate_at_word
            )

        return text
