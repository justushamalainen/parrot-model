"""
Shared utility functions for tool call helpers.

This module contains common functionality used by both Pydantic AI and LangChain
tool call helpers, following the DRY (Don't Repeat Yourself) principle.
"""

from __future__ import annotations

import uuid
from typing import Any


def encode_tool_call_in_prompt(tool_name: str, **parameters: Any) -> str:
    """
    Encode a tool call in the Parrot Model's [TOOL:name|param=value] syntax.

    This function creates a string that can be embedded in a prompt message
    to trigger tool calls when using the Parrot Model. This encoding is
    framework-agnostic and works with both Pydantic AI and LangChain.

    Args:
        tool_name: The name of the tool to call.
        **parameters: Keyword arguments representing tool parameters.

    Returns:
        str: The encoded tool call string in [TOOL:name|param=value] format.

    Example:
        >>> encode_tool_call_in_prompt("get_weather", city="London", units="metric")
        '[TOOL:get_weather|city=London|units=metric]'

        >>> encode_tool_call_in_prompt("simple_tool")
        '[TOOL:simple_tool|]'
    """
    if not parameters:
        return f"[TOOL:{tool_name}|]"

    # Sort parameters alphabetically for consistency
    param_parts = [f"{key}={value}" for key, value in sorted(parameters.items())]
    param_string = "|".join(param_parts)

    return f"[TOOL:{tool_name}|{param_string}]"


def generate_tool_call_id(tool_call_id: str | None = None) -> str:
    """
    Generate a tool call ID, using provided ID or creating a new UUID.

    This ensures consistent ID generation across both Pydantic AI and LangChain
    helpers. If an ID is provided, it's returned as-is. Otherwise, a new UUID4
    is generated.

    Args:
        tool_call_id: Optional pre-existing tool call ID. If None, generates new UUID.

    Returns:
        str: The tool call ID (either provided or newly generated).

    Example:
        >>> generate_tool_call_id("my-custom-id")
        'my-custom-id'

        >>> id1 = generate_tool_call_id()
        >>> id2 = generate_tool_call_id()
        >>> id1 != id2  # Each call generates unique ID
        True
    """
    return tool_call_id if tool_call_id is not None else str(uuid.uuid4())


__all__ = [
    "encode_tool_call_in_prompt",
    "generate_tool_call_id",
]
