"""
Core functionality tests for the Parrot Model.

This module tests the fundamental features of the Parrot Model including:
- Basic echo mode
- Text truncation (token and character based)
- Async generation
- Tool call parsing and handling
- Streaming (sync and async)
- Edge cases (empty messages, zero limits, malformed input)
"""

import pytest

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig


# ============================================================================
# Core Functionality Tests (3 tests)
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


@pytest.mark.parametrize(
    "config_kwargs,max_length,expected_start",
    [
        ({"max_tokens": 5}, 5, "This is a very long"),
        ({"max_chars": 20, "truncate_at_word": True}, 20, "This is a very"),
    ],
)
def test_text_truncation(sample_messages, config_kwargs, max_length, expected_start):
    """
    Test that messages are correctly truncated based on token or character limits.

    Validates both token-based and character-based truncation with appropriate limits.
    """
    config = ParrotConfig(**config_kwargs)
    model = ParrotModel(config=config)

    response = model.generate(sample_messages["long"])

    # Check appropriate limit is applied
    if "max_tokens" in config_kwargs:
        tokens = response.split()
        assert len(tokens) <= max_length
    else:
        assert len(response) <= max_length

    assert response.startswith(expected_start)


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
# Streaming Tests (2 tests)
# ============================================================================


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


# ============================================================================
# Edge Case Tests (4 tests)
# ============================================================================


def test_empty_message_handling(basic_model):
    """Test that empty messages are handled gracefully."""
    response = basic_model.generate("")
    assert response == ""
    assert len(basic_model.get_tool_calls()) == 0


def test_zero_limits():
    """Test that zero token/char limits return empty string."""
    config = ParrotConfig(max_tokens=0)
    model = ParrotModel(config=config)
    response = model.generate("Hello world")
    assert response == ""

    config = ParrotConfig(max_chars=0)
    model = ParrotModel(config=config)
    response = model.generate("Hello world")
    assert response == ""


def test_combined_token_and_char_limits(sample_messages):
    """Test that both token and character limits are applied correctly."""
    # Token limit applies first, then char limit
    config = ParrotConfig(max_tokens=10, max_chars=30)
    model = ParrotModel(config=config)
    response = model.generate(sample_messages["long"])

    # Should respect both limits
    assert len(response.split()) <= 10
    assert len(response) <= 30


def test_malformed_tool_call_syntax(basic_model):
    """Test that malformed tool call syntax is handled gracefully."""
    # Incomplete tool call - missing closing bracket
    response = basic_model.generate("Check [TOOL:incomplete")
    # Should treat as regular text since it doesn't match pattern
    assert "[TOOL:incomplete" in response

    # Empty parameters
    response = basic_model.generate("Call [TOOL:function|] please")
    # Parser should handle empty params gracefully
    tool_calls = basic_model.get_tool_calls()
    # Empty params might create a tool call with no parameters
    if tool_calls:
        assert tool_calls[0].name == "function"
        assert tool_calls[0].parameters == {}
