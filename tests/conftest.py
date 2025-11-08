"""
Pytest configuration and shared fixtures for Parrot Model tests.

This module defines pytest fixtures and configuration that are shared
across all test modules, including common test data, mock objects,
and setup/teardown helpers.
"""

import pytest

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

# Check for optional dependencies
try:
    import pydantic_ai  # noqa: F401

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage  # noqa: F401

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# Skip markers for optional dependencies
pytest.mark.skipif_no_pydantic_ai = pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed"
)

pytest.mark.skipif_no_langchain = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain-core not installed"
)


@pytest.fixture
def basic_model():
    """
    Fixture providing a ParrotModel with default configuration.

    Returns:
        ParrotModel: A model instance with default settings.
    """
    return ParrotModel()


@pytest.fixture
def configured_model():
    """
    Fixture providing a ParrotModel with custom configuration.

    Returns:
        ParrotModel: A model instance with custom settings for testing
        truncation, streaming delay, etc.
    """
    config = ParrotConfig(
        max_tokens=10,
        max_chars=50,
        truncate_at_word=True,
        stream_delay_ms=10,  # Faster for tests
        enable_tool_calls=True,
    )
    return ParrotModel(config=config)


@pytest.fixture
def model_without_tool_calls():
    """
    Fixture providing a ParrotModel with tool calls disabled.

    Returns:
        ParrotModel: A model instance with tool calls disabled.
    """
    config = ParrotConfig(enable_tool_calls=False)
    return ParrotModel(config=config)


@pytest.fixture
def sample_messages():
    """
    Fixture providing sample messages for testing.

    Returns:
        dict: Dictionary of sample messages for various test scenarios.
    """
    return {
        "simple": "Hello, world!",
        "long": "This is a very long message that will be used to test truncation functionality in both token-based and character-based modes.",  # noqa: E501
        "with_tool_call": "Get weather: [TOOL:get_weather|city=London|units=metric]",
        "multiple_tool_calls": "Compare [TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris]",  # noqa: E501
        "mixed_content": "First check [TOOL:search|query=Python] then send [TOOL:email|to=user@example.com]",  # noqa: E501
    }
