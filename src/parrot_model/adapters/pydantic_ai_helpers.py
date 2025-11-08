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

from typing import Any

from parrot_model.adapters._tool_call_utils import (
    encode_tool_call_in_prompt,
    generate_tool_call_id,
)

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
) -> ToolCallPart:
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

    # Create and return the ToolCallPart
    return ToolCallPart(
        tool_name=tool_name,
        args=parameters,
        tool_call_id=generate_tool_call_id(tool_call_id)
    )


__all__ = [
    "create_tool_call",
    "encode_tool_call_in_prompt",
]
