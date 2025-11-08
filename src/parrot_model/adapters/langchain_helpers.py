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
        >>> from langchain_core.messages import AIMessage
        >>>
        >>> # Create a single tool call
        >>> tool_call = create_tool_call("get_weather", city="London")
        >>> tool_call.name
        'get_weather'
        >>> tool_call.args
        {'city': 'London'}
        >>>
        >>> # Use in an AIMessage
        >>> message = AIMessage(content="Checking weather", tool_calls=[tool_call])
        >>>
        >>> # Create multiple tool calls with list comprehension
        >>> tool_calls = [
        ...     create_tool_call("get_weather", city="London"),
        ...     create_tool_call("get_time", timezone="UTC")
        ... ]
        >>> message = AIMessage(content="Processing", tool_calls=tool_calls)
    """
    _check_langchain_available()

    return ToolCall(
        name=tool_name,
        args=parameters,
        id=generate_tool_call_id(tool_call_id),
    )


__all__ = [
    "create_tool_call",
    "encode_tool_call_in_prompt",
]
