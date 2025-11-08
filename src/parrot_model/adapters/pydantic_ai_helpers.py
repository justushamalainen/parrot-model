"""
Helper functions for creating Pydantic AI tool calls.

This module provides convenient functions for creating tool call messages
that can be added to Pydantic AI message lists.

Example:
    >>> from parrot_model.adapters.pydantic_ai_helpers import create_tool_call
    >>> tool_call = create_tool_call("get_weather", city="London", units="metric")
    >>> # Add to your message request
    >>> from pydantic_ai.messages import ModelRequest
    >>> request = ModelRequest(parts=[tool_call])
"""

from __future__ import annotations

import uuid
from typing import Any

# Try to import Pydantic AI components
try:
    from pydantic_ai.messages import ToolCallPart

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    ToolCallPart = None  # type: ignore


def _check_pydantic_ai_available():
    """Check if pydantic-ai is installed and raise helpful error if not."""
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "Pydantic AI is not installed. "
            "Please install it with: pip install pydantic-ai\n"
            "Or install parrot-model with the pydantic-ai extra: "
            "pip install parrot-model[pydantic-ai]"
        )


def create_tool_call(
    tool_name: str,
    tool_call_id: str | None = None,
    **parameters: Any
) -> "ToolCallPart":
    """
    Create a Pydantic AI ToolCallPart for use in messages.

    This helper function creates a properly formatted tool call that can be
    added to a Pydantic AI ModelRequest or used in agent workflows.

    Args:
        tool_name: The name of the tool/function to call.
        tool_call_id: Optional unique identifier for this tool call.
                     If None, a UUID4 will be generated.
        **parameters: Keyword arguments representing the tool parameters.

    Returns:
        ToolCallPart: A Pydantic AI ToolCallPart object that can be added to messages.

    Raises:
        ImportError: If pydantic-ai is not installed.

    Example:
        >>> from parrot_model.adapters.pydantic_ai_helpers import create_tool_call
        >>>
        >>> # Create a simple tool call
        >>> tool_call = create_tool_call("get_weather", city="London")
        >>>
        >>> # Create with multiple parameters
        >>> tool_call = create_tool_call(
        ...     "search_database",
        ...     query="users",
        ...     limit=10,
        ...     offset=0
        ... )
        >>>
        >>> # Create with custom ID
        >>> tool_call = create_tool_call(
        ...     "calculate",
        ...     tool_call_id="calc-001",
        ...     operation="add",
        ...     x=5,
        ...     y=3
        ... )
        >>>
        >>> # Use in a ModelRequest
        >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
        >>> request = ModelRequest(parts=[
        ...     UserPromptPart(content="Please check the weather"),
        ...     create_tool_call("get_weather", city="London")
        ... ])
    """
    _check_pydantic_ai_available()

    # Generate ID if not provided
    if tool_call_id is None:
        tool_call_id = str(uuid.uuid4())

    # Create and return the ToolCallPart
    return ToolCallPart(
        tool_name=tool_name,
        args=parameters,
        tool_call_id=tool_call_id
    )


def create_tool_calls(
    *calls: tuple[str, dict[str, Any]]
) -> list["ToolCallPart"]:
    """
    Create multiple Pydantic AI ToolCallPart objects at once.

    This is a convenience function for creating multiple tool calls in a single
    call. Each tool call is specified as a tuple of (tool_name, parameters_dict).

    Args:
        *calls: Variable number of tuples, each containing:
               - tool_name (str): The name of the tool
               - parameters (dict): Dictionary of parameters for the tool

    Returns:
        list[ToolCallPart]: List of ToolCallPart objects.

    Raises:
        ImportError: If pydantic-ai is not installed.

    Example:
        >>> from parrot_model.adapters.pydantic_ai_helpers import create_tool_calls
        >>>
        >>> # Create multiple tool calls at once
        >>> tool_calls = create_tool_calls(
        ...     ("get_weather", {"city": "London"}),
        ...     ("get_weather", {"city": "Paris"}),
        ...     ("get_time", {"timezone": "UTC"})
        ... )
        >>>
        >>> # Use in a ModelRequest
        >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
        >>> request = ModelRequest(parts=[
        ...     UserPromptPart(content="Check weather in multiple cities"),
        ...     *tool_calls
        ... ])
    """
    _check_pydantic_ai_available()

    tool_call_parts = []
    for tool_name, parameters in calls:
        tool_call_parts.append(create_tool_call(tool_name, **parameters))

    return tool_call_parts


def encode_tool_call_in_prompt(tool_name: str, **parameters: Any) -> str:
    """
    Encode a tool call in the Parrot Model's [TOOL:name|param=value] syntax.

    This function creates a string that can be embedded in a prompt message
    to trigger tool calls when using the Parrot Model with Pydantic AI.

    Args:
        tool_name: The name of the tool to call.
        **parameters: Keyword arguments representing tool parameters.

    Returns:
        str: The encoded tool call string in [TOOL:name|param=value] format.

    Example:
        >>> from parrot_model.adapters.pydantic_ai_helpers import encode_tool_call_in_prompt
        >>>
        >>> # Create a tool call encoding
        >>> encoding = encode_tool_call_in_prompt("get_weather", city="London", units="metric")
        >>> encoding
        '[TOOL:get_weather|city=London|units=metric]'
        >>>
        >>> # Use in a message
        >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
        >>> request = ModelRequest(parts=[
        ...     UserPromptPart(content=f"Check {encoding} today")
        ... ])
        >>> # Result: "Check [TOOL:get_weather|city=London|units=metric] today"
    """
    if not parameters:
        return f"[TOOL:{tool_name}|]"

    # Sort parameters for consistency
    param_parts = [f"{key}={value}" for key, value in sorted(parameters.items())]
    param_string = "|".join(param_parts)

    return f"[TOOL:{tool_name}|{param_string}]"


__all__ = [
    "create_tool_call",
    "create_tool_calls",
    "encode_tool_call_in_prompt",
]
