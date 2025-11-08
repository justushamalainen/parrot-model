"""
Core functionality tests for the Parrot Model.

This module tests the fundamental features of the Parrot Model including:
- Basic echo mode
- Text truncation (token and character based)
- Async generation
- Configuration handling
- Tool call parsing and handling
- Streaming (sync and async)
"""

import pytest

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig


# ============================================================================
# Core Functionality Tests (5 tests)
# ============================================================================


def test_basic_echo(basic_model, sample_messages):
    """
    Test that the model correctly echoes back simple messages.

    Validates the fundamental echo mode behavior where the model returns
    the input message unchanged.
    """
    response = basic_model.generate(sample_messages["simple"])
    assert response == sample_messages["simple"]
    assert len(basic_model.get_tool_calls()) == 0


def test_token_based_truncation(sample_messages):
    """
    Test that messages are correctly truncated based on token count.

    Validates that when max_tokens is set, the response is limited to
    approximately that many whitespace-delimited tokens.
    """
    config = ParrotConfig(max_tokens=5)
    model = ParrotModel(config=config)

    response = model.generate(sample_messages["long"])

    # Should be truncated to approximately 5 tokens
    tokens = response.split()
    assert len(tokens) <= 5
    assert response.startswith("This is a very long")


def test_character_based_truncation(sample_messages):
    """
    Test that messages are correctly truncated based on character count.

    Validates that when max_chars is set, the response is limited to
    that many characters (or fewer if truncate_at_word is enabled).
    """
    config = ParrotConfig(max_chars=20, truncate_at_word=True)
    model = ParrotModel(config=config)

    response = model.generate(sample_messages["long"])

    # Should be truncated to max 20 chars
    assert len(response) <= 20
    assert response.startswith("This is a very")


@pytest.mark.asyncio
async def test_async_generation(basic_model, sample_messages):
    """
    Test asynchronous message generation.

    Validates that the async generate method works correctly and returns
    the same results as the synchronous version.
    """
    response = await basic_model.agenerate(sample_messages["simple"])
    assert response == sample_messages["simple"]

    # Test with tool calls
    response = await basic_model.agenerate(sample_messages["with_tool_call"])
    assert "Get weather:" in response
    assert len(basic_model.get_tool_calls()) == 1


def test_configuration_handling():
    """
    Test that configuration is properly stored and applied.

    Validates that custom configuration settings are correctly initialized
    and used by the model.
    """
    config = ParrotConfig(
        mode="echo",
        max_tokens=100,
        max_chars=500,
        truncate_at_word=False,
        stream_delay_ms=25,
        enable_tool_calls=True,
    )
    model = ParrotModel(config=config)

    assert model.config.mode == "echo"
    assert model.config.max_tokens == 100
    assert model.config.max_chars == 500
    assert model.config.truncate_at_word is False
    assert model.config.stream_delay_ms == 25
    assert model.config.enable_tool_calls is True


# ============================================================================
# Tool Call Tests (3 tests)
# ============================================================================


def test_single_tool_call_parsing(basic_model, sample_messages):
    """
    Test parsing of a single tool call from a message.

    Validates that tool call syntax is correctly detected, parsed, and
    removed from the response text. The parsed tool call should be
    accessible via get_tool_calls().
    """
    response = basic_model.generate(sample_messages["with_tool_call"])

    # Tool call syntax should be removed from response
    assert "[TOOL:" not in response
    assert response == "Get weather:"

    # Tool call should be parsed and available
    tool_calls = basic_model.get_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].parameters["city"] == "London"
    assert tool_calls[0].parameters["units"] == "metric"
    assert tool_calls[0].id is not None  # Should have a UUID


def test_multiple_tool_calls(basic_model, sample_messages):
    """
    Test parsing of multiple tool calls in a single message.

    Validates that multiple tool calls can be detected and parsed from
    a single message, and that each has its own unique ID.
    """
    response = basic_model.generate(sample_messages["multiple_tool_calls"])

    # Tool call syntax should be removed
    assert "[TOOL:" not in response
    assert response == "Compare and"

    # Both tool calls should be parsed
    tool_calls = basic_model.get_tool_calls()
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].parameters["city"] == "London"
    assert tool_calls[1].name == "get_weather"
    assert tool_calls[1].parameters["city"] == "Paris"

    # Each should have a unique ID
    assert tool_calls[0].id != tool_calls[1].id


def test_tool_calls_disabled(model_without_tool_calls, sample_messages):
    """
    Test that tool calls can be disabled via configuration.

    Validates that when enable_tool_calls is False, tool call syntax
    is treated as regular text and not parsed.
    """
    response = model_without_tool_calls.generate(sample_messages["with_tool_call"])

    # Tool call syntax should remain in response
    assert "[TOOL:get_weather|city=London|units=metric]" in response
    assert response == sample_messages["with_tool_call"]

    # No tool calls should be parsed
    tool_calls = model_without_tool_calls.get_tool_calls()
    assert len(tool_calls) == 0


# ============================================================================
# Streaming Tests (3 tests)
# ============================================================================


def test_sync_streaming(configured_model, sample_messages):
    """
    Test synchronous streaming of responses.

    Validates that the stream() method yields chunks of the response
    and that they can be reassembled into the complete message.
    """
    chunks = list(configured_model.stream(sample_messages["simple"], chunk_size=5))

    # Should have multiple chunks
    assert len(chunks) > 1

    # Chunks should reassemble to original
    complete = "".join(chunks)
    assert complete == sample_messages["simple"]


@pytest.mark.asyncio
async def test_async_streaming(configured_model, sample_messages):
    """
    Test asynchronous streaming of responses.

    Validates that the astream() method yields chunks asynchronously
    and that they can be reassembled into the complete message.
    """
    chunks = []
    async for chunk in configured_model.astream(sample_messages["simple"], chunk_size=5):
        chunks.append(chunk)

    # Should have multiple chunks
    assert len(chunks) > 1

    # Chunks should reassemble to original
    complete = "".join(chunks)
    assert complete == sample_messages["simple"]


def test_streaming_with_text_limits(sample_messages):
    """
    Test streaming with text truncation applied.

    Validates that when text limits are configured, streaming properly
    respects those limits and streams only the truncated text.
    """
    config = ParrotConfig(max_chars=20, truncate_at_word=True, stream_delay_ms=5)
    model = ParrotModel(config=config)

    chunks = list(model.stream(sample_messages["long"], chunk_size=3))

    # Chunks should reassemble to truncated text
    complete = "".join(chunks)
    assert len(complete) <= 20
    assert complete.startswith("This is a very")
