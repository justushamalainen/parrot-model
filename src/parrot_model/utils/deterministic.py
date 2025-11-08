"""
Utilities for deterministic ID and timestamp generation.

This module provides functions for generating deterministic IDs and timestamps
to ensure that the Parrot Model produces consistent, repeatable results for
the same inputs - a core feature for testing and development.
"""

import hashlib
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parrot_model.core.config import ParrotConfig


def generate_deterministic_id(
    *inputs: Any, prefix: str = "", length: int = 8
) -> str:
    """
    Generate a deterministic ID based on input content.

    Creates a consistent ID by hashing the input content. The same inputs
    will always produce the same ID, ensuring deterministic behavior.

    Args:
        *inputs: Variable number of inputs to hash. Each input is converted
                to a string and combined for hashing.
        prefix: Optional prefix to add before the hash. Default is empty string.
        length: Length of the hash portion of the ID. Default is 8 characters.

    Returns:
        A deterministic ID string combining the prefix and content hash.

    Example:
        >>> generate_deterministic_id("hello", "world", prefix="msg_")
        'msg_fc3ff98e'
        >>> # Same inputs always produce same ID
        >>> id1 = generate_deterministic_id("test", prefix="call_")
        >>> id2 = generate_deterministic_id("test", prefix="call_")
        >>> id1 == id2
        True
    """
    # Combine all inputs into a single string
    combined = "|".join(str(inp) for inp in inputs)

    # Create MD5 hash of the combined string
    # MD5 is fine here since we're not using it for security
    hash_obj = hashlib.md5(combined.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()[:length]

    # Combine prefix with hash
    return f"{prefix}{hash_hex}"


def generate_tool_call_id(*inputs: Any) -> str:
    """
    Generate a deterministic tool call ID.

    Creates a consistent tool call ID in UUID-like format based on content.
    The ID format mimics UUID4 but is deterministic.

    Args:
        *inputs: Variable number of inputs to hash (typically tool name and parameters).

    Returns:
        A deterministic UUID-like string.

    Example:
        >>> generate_tool_call_id("get_weather", "city=London")
        'a7ffc6f8-beef-4cc0-825c-2fcf932b1f01'
        >>> # Same inputs produce same ID
        >>> id1 = generate_tool_call_id("search", "q=test")
        >>> id2 = generate_tool_call_id("search", "q=test")
        >>> id1 == id2
        True
    """
    # Combine all inputs
    combined = "|".join(str(inp) for inp in inputs)

    # Create MD5 hash
    hash_obj = hashlib.md5(combined.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    # Format as UUID-like string (8-4-4-4-12 format)
    # Take different parts of the hash to create the UUID format
    return (
        f"{hash_hex[:8]}-"
        f"{hash_hex[8:12]}-"
        f"{hash_hex[12:16]}-"
        f"{hash_hex[16:20]}-"
        f"{hash_hex[20:32]}"
    )


def get_deterministic_timestamp(base_time: int = 1700000000) -> int:
    """
    Return a fixed, deterministic timestamp.

    Returns a constant timestamp for deterministic behavior. This ensures
    that timestamp fields don't cause variation in test outputs.

    Args:
        base_time: The fixed timestamp to return. Default is 1700000000
                  (November 14, 2023 22:13:20 UTC).

    Returns:
        The fixed timestamp as an integer.

    Example:
        >>> get_deterministic_timestamp()
        1700000000
        >>> # Always returns the same value
        >>> t1 = get_deterministic_timestamp()
        >>> t2 = get_deterministic_timestamp()
        >>> t1 == t2
        True
    """
    return base_time


def hash_content(*parts: Any) -> str:
    """
    Create a deterministic hash of content parts.

    Combines multiple content parts and returns their MD5 hash.
    Useful for creating content-based identifiers.

    Args:
        *parts: Variable number of content parts to hash.

    Returns:
        Hexadecimal hash string.

    Example:
        >>> hash_content("hello", "world")
        'fc3ff98e8c6a0d3087d515c0473f8677'
        >>> # Same content produces same hash
        >>> h1 = hash_content("test", 123)
        >>> h2 = hash_content("test", 123)
        >>> h1 == h2
        True
    """
    combined = "|".join(str(part) for part in parts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# Config-aware helper functions


def get_timestamp_for_config(config: "ParrotConfig") -> int:
    """
    Get timestamp based on config deterministic setting.

    Uses deterministic timestamp when config.deterministic is True,
    otherwise returns current timestamp.

    Args:
        config: ParrotConfig instance to check deterministic setting.

    Returns:
        Fixed timestamp if deterministic, current timestamp otherwise.

    Example:
        >>> from parrot_model.core.config import ParrotConfig
        >>> config = ParrotConfig(deterministic=True)
        >>> get_timestamp_for_config(config)
        1700000000
        >>> config = ParrotConfig(deterministic=False)
        >>> get_timestamp_for_config(config)  # Returns current time
        1733...
    """
    if config.deterministic:
        return get_deterministic_timestamp()
    else:
        return int(time.time())


def generate_id_for_config(
    config: "ParrotConfig",
    *inputs: Any,
    prefix: str = "",
    length: int = 8,
) -> str:
    """
    Generate ID based on config deterministic setting.

    Uses content-based deterministic ID when config.deterministic is True,
    otherwise generates random UUID-based ID.

    Args:
        config: ParrotConfig instance to check deterministic setting.
        *inputs: Variable number of inputs for deterministic hash.
        prefix: Prefix to add before the ID.
        length: Length of the hash/UUID portion.

    Returns:
        Deterministic ID if configured, random ID otherwise.

    Example:
        >>> from parrot_model.core.config import ParrotConfig
        >>> config = ParrotConfig(deterministic=True)
        >>> generate_id_for_config(config, "test", prefix="msg_", length=8)
        'msg_098f6bcd'
        >>> # With random IDs
        >>> config = ParrotConfig(deterministic=False)
        >>> id1 = generate_id_for_config(config, "test", prefix="msg_", length=8)
        >>> id2 = generate_id_for_config(config, "test", prefix="msg_", length=8)
        >>> id1 != id2  # Random IDs differ
        True
    """
    if config.deterministic:
        return generate_deterministic_id(*inputs, prefix=prefix, length=length)
    else:
        return f"{prefix}{uuid.uuid4().hex[:length]}"
