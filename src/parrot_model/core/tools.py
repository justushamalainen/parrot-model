"""
Tool call handling for the Parrot Model.

This module provides functionality for parsing tool call encodings
from messages using the [TOOL:name|param=value] syntax and generating
appropriate tool call responses for different frameworks.
"""

import re
import uuid
from dataclasses import dataclass

from parrot_model.utils.deterministic import generate_tool_call_id
from parrot_model.utils.parser import parse_tool_parameters


@dataclass
class ToolCall:
    """
    Represents a parsed tool call from a message.

    A tool call contains the tool name, its parameters, and a unique identifier.
    Tool calls are extracted from messages using the [TOOL:name|param=value] syntax.

    Attributes:
        name: The name of the tool to be called.
        parameters: Dictionary of parameter names to their string values.
        id: Unique identifier for this tool call (UUID4).

    Example:
        >>> tool_call = ToolCall(
        ...     name="get_weather",
        ...     parameters={"city": "London", "units": "metric"},
        ...     id="550e8400-e29b-41d4-a716-446655440000"
        ... )
        >>> tool_call.name
        'get_weather'
        >>> tool_call.parameters["city"]
        'London'
    """

    name: str
    parameters: dict[str, str]
    id: str


class ToolCallParser:
    """
    Parser for extracting tool calls from messages.

    This class provides functionality to parse tool call encodings from messages,
    remove tool call syntax from messages, and encode tool calls back into the
    standard format.

    The parser uses a regular expression pattern to match tool calls in the format:
    [TOOL:function_name|param1=value1|param2=value2]

    Attributes:
        pattern: Compiled regular expression pattern for matching tool calls.

    Example:
        >>> parser = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]")
        >>> message = "Check [TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris]"
        >>> tool_calls = parser.parse(message)
        >>> len(tool_calls)
        2
        >>> tool_calls[0].name
        'get_weather'
        >>> tool_calls[0].parameters
        {'city': 'London'}
    """

    def __init__(self, pattern: str, deterministic: bool = True):
        """
        Initialize the ToolCallParser with a regex pattern.

        Args:
            pattern: Regular expression pattern to match tool calls.
                    Should have two capture groups: (tool_name) and (parameters).
            deterministic: Whether to generate deterministic IDs. When True (default),
                          IDs are generated based on content hash. When False, random
                          UUIDs are used.

        Example:
            >>> parser = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]")
            >>> parser_random = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]", deterministic=False)
        """
        self.pattern = re.compile(pattern)
        self.deterministic = deterministic

    def parse(self, message: str) -> list[ToolCall]:
        """
        Extract all tool calls from a message.

        Parses the message to find all tool call encodings and converts them
        into ToolCall objects with generated UUIDs.

        Args:
            message: The message containing potential tool call encodings.

        Returns:
            List of ToolCall objects found in the message. Empty list if none found.

        Example:
            >>> parser = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]")
            >>> message = "Get [TOOL:get_weather|city=London|units=metric]"
            >>> tool_calls = parser.parse(message)
            >>> tool_calls[0].name
            'get_weather'
            >>> tool_calls[0].parameters
            {'city': 'London', 'units': 'metric'}
        """
        tool_calls = []
        matches = self.pattern.finditer(message)

        for match in matches:
            tool_name = match.group(1)
            param_string = match.group(2)

            # Parse the parameters
            parameters = parse_tool_parameters(param_string)

            # Generate ID based on deterministic mode
            if self.deterministic:
                # Create deterministic ID from tool name and parameters
                # Sort parameters for consistency
                sorted_params = sorted(parameters.items())
                param_repr = "|".join(f"{k}={v}" for k, v in sorted_params)
                tool_id = generate_tool_call_id(tool_name, param_repr)
            else:
                # Use random UUID
                tool_id = str(uuid.uuid4())

            # Create ToolCall with ID
            tool_call = ToolCall(
                name=tool_name,
                parameters=parameters,
                id=tool_id
            )
            tool_calls.append(tool_call)

        return tool_calls

    def remove_from_message(self, message: str) -> str:
        """
        Remove tool call syntax from a message, leaving clean text.

        Removes all [TOOL:...] encodings from the message, leaving only the
        surrounding text. Multiple spaces are collapsed to single spaces,
        and leading/trailing whitespace is trimmed.

        Args:
            message: The message potentially containing tool call encodings.

        Returns:
            The message with all tool call encodings removed and whitespace normalized.

        Example:
            >>> parser = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]")
            >>> message = "Check [TOOL:get_weather|city=London] today"
            >>> parser.remove_from_message(message)
            'Check today'
            >>> parser.remove_from_message("Just [TOOL:search|query=test] it")
            'Just it'
        """
        # Remove all tool call encodings
        clean_message = self.pattern.sub("", message)

        # Normalize whitespace: collapse multiple spaces and trim
        clean_message = re.sub(r"\s+", " ", clean_message).strip()

        return clean_message

    @staticmethod
    def encode(tool_name: str, **params) -> str:
        """
        Create a tool call encoding string.

        Encodes a tool call into the standard [TOOL:name|param=value] format.
        Parameters are sorted alphabetically for consistency.

        Args:
            tool_name: The name of the tool to call.
            **params: Keyword arguments representing tool parameters.

        Returns:
            The encoded tool call string.

        Example:
            >>> ToolCallParser.encode("get_weather", city="London", units="metric")
            '[TOOL:get_weather|city=London|units=metric]'
            >>> ToolCallParser.encode("search_db", query="users", limit="10")
            '[TOOL:search_db|limit=10|query=users]'
        """
        if not params:
            # No parameters case
            return f"[TOOL:{tool_name}|]"

        # Sort parameters for consistency
        param_parts = [f"{key}={value}" for key, value in sorted(params.items())]
        param_string = "|".join(param_parts)

        return f"[TOOL:{tool_name}|{param_string}]"
