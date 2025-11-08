"""
Helper functions for creating LangChain/LangGraph tool calls.

This module provides convenient functions for creating tool call messages
that can be added to LangChain message lists or used in LangGraph workflows.

Example:
    >>> from parrot_model.adapters.langchain_helpers import create_tool_call_message
    >>> from langchain_core.messages import HumanMessage
    >>>
    >>> # Create a message with a tool call
    >>> message = create_tool_call_message("get_weather", city="London")
    >>> # Use in a chain or graph
    >>> messages = [HumanMessage(content="Hello"), message]
"""

from __future__ import annotations

import json
from typing import Any

from parrot_model.adapters._tool_call_utils import (
    encode_tool_call_in_prompt,
    generate_tool_call_id,
)

# Try to import LangChain components
try:
    from langchain_core.messages import AIMessage
    from langchain_core.messages.tool import ToolCall

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    AIMessage = None  # type: ignore
    ToolCall = None  # type: ignore


def _check_langchain_available():
    """Check if langchain-core is installed and raise helpful error if not."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. "
            "Please install it with: pip install langchain-core\n"
            "Or install parrot-model with the langchain extra: "
            "pip install parrot-model[langchain]"
        )


def create_tool_call(
    tool_name: str,
    tool_call_id: str | None = None,
    **parameters: Any
) -> "ToolCall":
    """
    Create a LangChain ToolCall object.

    This creates a ToolCall object using LangChain's native ToolCall class,
    which can be added to a message's tool_calls list.

    Args:
        tool_name: The name of the tool/function to call.
        tool_call_id: Optional unique identifier for this tool call.
                     If None, a UUID4 will be generated.
        **parameters: Keyword arguments representing the tool parameters.

    Returns:
        ToolCall: A LangChain ToolCall object.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from parrot_model.adapters.langchain_helpers import create_tool_call
        >>>
        >>> tool_call = create_tool_call("get_weather", city="London")
        >>> tool_call.name
        'get_weather'
        >>> tool_call.args
        {'city': 'London'}
    """
    _check_langchain_available()

    return ToolCall(
        name=tool_name,
        args=parameters,
        id=generate_tool_call_id(tool_call_id),
    )


def create_tool_calls(
    *calls: tuple[str, dict[str, Any]]
) -> list["ToolCall"]:
    """
    Create multiple LangChain ToolCall objects at once.

    This is a convenience function for creating multiple tool calls in a single
    call using LangChain's native ToolCall class. Each tool call is specified
    as a tuple of (tool_name, parameters_dict).

    Args:
        *calls: Variable number of tuples, each containing:
               - tool_name (str): The name of the tool
               - parameters (dict): Dictionary of parameters for the tool

    Returns:
        list[ToolCall]: List of LangChain ToolCall objects.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from parrot_model.adapters.langchain_helpers import create_tool_calls
        >>>
        >>> # Create multiple tool calls at once
        >>> tool_calls = create_tool_calls(
        ...     ("get_weather", {"city": "London"}),
        ...     ("get_weather", {"city": "Paris"}),
        ...     ("get_time", {"timezone": "UTC"})
        ... )
        >>>
        >>> # Use in an AIMessage
        >>> from langchain_core.messages import AIMessage
        >>> message = AIMessage(content="Processing requests", tool_calls=tool_calls)
    """
    _check_langchain_available()

    return [
        create_tool_call(tool_name, **parameters)
        for tool_name, parameters in calls
    ]


def create_tool_call_dict(
    tool_name: str,
    tool_call_id: str | None = None,
    **parameters: Any
) -> dict[str, Any]:
    """
    Create a tool call dictionary in OpenAI format for LangChain.

    This creates a tool call dictionary that can be added to a message's
    additional_kwargs["tool_calls"] list. For most use cases, prefer using
    create_tool_call() which returns a proper ToolCall object.

    Args:
        tool_name: The name of the tool/function to call.
        tool_call_id: Optional unique identifier for this tool call.
                     If None, a UUID4 will be generated.
        **parameters: Keyword arguments representing the tool parameters.

    Returns:
        dict: A tool call dictionary in OpenAI format.

    Example:
        >>> from parrot_model.adapters.langchain_helpers import create_tool_call_dict
        >>>
        >>> tool_call = create_tool_call_dict("get_weather", city="London")
        >>> tool_call
        {
            'id': '...',
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'arguments': '{"city": "London"}'
            }
        }
    """
    return {
        "id": generate_tool_call_id(tool_call_id),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(parameters),
        },
    }


def create_tool_call_message(
    tool_name: str,
    content: str = "",
    tool_call_id: str | None = None,
    **parameters: Any
) -> "AIMessage":
    """
    Create a LangChain AIMessage with a tool call.

    This helper function creates a properly formatted AI message with a tool call
    using LangChain's native ToolCall class. The message can be added to a
    LangChain message list or used in LangGraph workflows.

    Args:
        tool_name: The name of the tool/function to call.
        content: Optional text content for the message. Defaults to empty string.
        tool_call_id: Optional unique identifier for this tool call.
                     If None, a UUID4 will be generated.
        **parameters: Keyword arguments representing the tool parameters.

    Returns:
        AIMessage: A LangChain AIMessage with the tool call.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from parrot_model.adapters.langchain_helpers import create_tool_call_message
        >>> from langchain_core.messages import HumanMessage
        >>>
        >>> # Create a simple tool call message
        >>> message = create_tool_call_message("get_weather", city="London")
        >>>
        >>> # Create with content
        >>> message = create_tool_call_message(
        ...     "get_weather",
        ...     content="Let me check the weather for you",
        ...     city="London",
        ...     units="metric"
        ... )
        >>>
        >>> # Use in a message list
        >>> messages = [
        ...     HumanMessage(content="What's the weather in London?"),
        ...     create_tool_call_message("get_weather", city="London")
        ... ]
        >>>
        >>> # Use with ParrotChatModel
        >>> from parrot_model.adapters import ParrotChatModel
        >>> model = ParrotChatModel()
        >>> # Process the message list
    """
    _check_langchain_available()

    # Create the tool call using LangChain's ToolCall class
    tool_call = create_tool_call(tool_name, tool_call_id, **parameters)

    # Create AIMessage with tool call
    return AIMessage(
        content=content,
        tool_calls=[tool_call]
    )


def create_multi_tool_call_message(
    *calls: tuple[str, dict[str, Any]],
    content: str = ""
) -> "AIMessage":
    """
    Create a LangChain AIMessage with multiple tool calls.

    This is a convenience function for creating a message with multiple tool calls
    using LangChain's native ToolCall class. Each tool call is specified as a
    tuple of (tool_name, parameters_dict).

    Args:
        *calls: Variable number of tuples, each containing:
               - tool_name (str): The name of the tool
               - parameters (dict): Dictionary of parameters for the tool
        content: Optional text content for the message. Defaults to empty string.

    Returns:
        AIMessage: A LangChain AIMessage with multiple tool calls.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from parrot_model.adapters.langchain_helpers import create_multi_tool_call_message
        >>>
        >>> # Create message with multiple tool calls
        >>> message = create_multi_tool_call_message(
        ...     ("get_weather", {"city": "London"}),
        ...     ("get_weather", {"city": "Paris"}),
        ...     ("get_time", {"timezone": "UTC"}),
        ...     content="Checking multiple locations"
        ... )
        >>>
        >>> # Access the tool calls
        >>> tool_calls = message.tool_calls
        >>> len(tool_calls)
        3
    """
    _check_langchain_available()

    # Create AIMessage with all tool calls (reuse create_tool_calls for DRY)
    return AIMessage(
        content=content,
        tool_calls=create_tool_calls(*calls)
    )

__all__ = [
    "create_tool_call",
    "create_tool_calls",
    "create_tool_call_dict",
    "create_tool_call_message",
    "create_multi_tool_call_message",
    "encode_tool_call_in_prompt",
]
