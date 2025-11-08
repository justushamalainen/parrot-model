"""
Message parsing utilities for the Parrot Model.

This module provides utilities for parsing and processing messages,
including extracting tool call encodings and other special syntax
from user messages.
"""

import re


def parse_tool_parameters(param_string: str) -> dict[str, str]:
    """
    Parse tool parameters from a pipe-delimited string.

    Parses parameter strings in the format "key=value|key2=value2" into a
    dictionary. Handles edge cases including empty values, special characters,
    and quoted values.

    Args:
        param_string: The parameter string to parse (e.g., "city=London|units=metric").

    Returns:
        Dictionary mapping parameter names to their string values.
        Empty dict if param_string is empty or contains no valid parameters.

    Example:
        >>> parse_tool_parameters("city=London|units=metric")
        {'city': 'London', 'units': 'metric'}
        >>> parse_tool_parameters("query=users|limit=10")
        {'query': 'users', 'limit': '10'}
        >>> parse_tool_parameters("name=John Doe|age=30")
        {'name': 'John Doe', 'age': '30'}
        >>> parse_tool_parameters("")
        {}

    Notes:
        - Whitespace around parameter names and values is stripped
        - Empty values are preserved (e.g., "key=" becomes {'key': ''})
        - Parameters without '=' are skipped
        - Quoted values are supported (quotes are preserved in the value)
    """
    if not param_string or param_string.strip() == "":
        return {}

    parameters = {}

    # Split by pipe character
    param_parts = param_string.split("|")

    for part in param_parts:
        part = part.strip()
        if not part:
            continue

        # Check if this part contains an '=' sign
        if "=" not in part:
            # Skip invalid parameters
            continue

        # Split on the first '=' only (in case value contains '=')
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()

        if key:  # Only add if key is non-empty
            parameters[key] = value

    return parameters
