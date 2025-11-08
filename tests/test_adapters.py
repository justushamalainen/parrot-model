"""
Adapter integration tests for the Parrot Model.

This module tests the framework-specific adapters including:
- Pydantic AI adapter basic usage and tool call handling
- LangChain adapter basic usage and streaming

Tests are conditional and skip if the respective dependencies are not installed.
"""

import pytest

from parrot_model.core.config import ParrotConfig

# Check for optional dependencies at module level
try:
    import pydantic_ai  # noqa: F401
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================================
# Pydantic AI Adapter Tests (2 tests)
# ============================================================================


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")
@pytest.mark.asyncio
async def test_pydantic_ai_adapter_basic():
    """
    Test basic usage of the Pydantic AI adapter.

    Validates that the ParrotPydanticModel can be initialized and used
    to generate responses from Pydantic AI messages. Ensures proper
    integration with the Pydantic AI interface.
    """
    from parrot_model.adapters.pydantic_ai import ParrotPydanticModel

    # Create model with custom config
    model = ParrotPydanticModel(
        model_name="test-parrot", config=ParrotConfig(max_tokens=50)
    )

    # Test basic properties
    assert model.model_name == "test-parrot"
    assert model.system == "parrot"
    assert model.parrot_model is not None

    # Create a simple message
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello, world!")])]

    # Make a request
    response = await model.request(messages, None, {})

    # Verify response structure
    assert response is not None
    assert hasattr(response, "parts")
    assert len(response.parts) > 0
    assert hasattr(response, "model_name")
    assert response.model_name == "parrot-echo"

    # Check content - should echo back
    text_parts = [p for p in response.parts if hasattr(p, "content")]
    assert len(text_parts) > 0
    assert "Hello, world!" in str(text_parts[0].content)


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")
@pytest.mark.asyncio
async def test_pydantic_ai_adapter_with_tool_calls():
    """
    Test Pydantic AI adapter with tool call handling.

    Validates that tool calls embedded in messages are properly parsed
    and converted to Pydantic AI's ToolCallPart format. This ensures
    the adapter correctly bridges between Parrot Model's tool call syntax
    and Pydantic AI's tool call representation.
    """
    from pydantic_ai.messages import ToolCallPart

    from parrot_model.adapters.pydantic_ai import ParrotPydanticModel

    # Create model with tool calls enabled
    model = ParrotPydanticModel(config=ParrotConfig(enable_tool_calls=True))

    # Message with tool call syntax
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="Check weather: [TOOL:get_weather|city=London|units=metric]"
                )
            ]
        )
    ]

    response = await model.request(messages, None, {})

    # Should have response parts
    assert len(response.parts) > 0

    # Check for tool call part
    tool_parts = [p for p in response.parts if isinstance(p, ToolCallPart)]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_name == "get_weather"
    assert tool_parts[0].args == {"city": "London", "units": "metric"}
    assert tool_parts[0].tool_call_id is not None

    # Text should not contain the tool syntax
    text_parts = [p for p in response.parts if hasattr(p, "content")]
    if text_parts:
        assert "[TOOL:" not in str(text_parts[0].content)


# ============================================================================
# LangChain Adapter Tests (2 tests)
# ============================================================================


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_langchain_adapter_basic():
    """
    Test basic usage of the LangChain adapter.

    Validates that the ParrotChatModel can be initialized and used
    to generate responses from LangChain messages. Tests both single
    and multiple message handling, system message filtering, and tool calls.
    """
    from parrot_model.adapters.langchain import ParrotChatModel

    # Create model with custom config
    config = ParrotConfig(max_tokens=100, echo_system_messages=False)
    model = ParrotChatModel(parrot_config=config)

    # Test with single message
    messages = [HumanMessage(content="Hello, world!")]
    response = model.invoke(messages)

    assert response.content == "Hello, world!"

    # Test with multiple messages (including system)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2?"),
    ]
    response = model.invoke(messages)

    # Should only echo the user message (system filtered out by default)
    assert "What is 2+2?" in response.content
    assert "You are a helpful assistant" not in response.content

    # Test tool call handling
    messages = [HumanMessage(content="Get [TOOL:search|query=Python]")]
    response = model.invoke(messages)

    # Should have tool calls in additional_kwargs
    assert "tool_calls" in response.additional_kwargs
    tool_calls = response.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_langchain_adapter_streaming():
    """
    Test streaming functionality of the LangChain adapter.

    Validates that the adapter properly supports LangChain's streaming
    interface, yielding chunks that reassemble into the complete response.
    """
    from langchain_core.messages import AIMessageChunk

    from parrot_model.adapters.langchain import ParrotChatModel

    # Create model with faster streaming for tests
    config = ParrotConfig(stream_delay_ms=5)
    model = ParrotChatModel(parrot_config=config)

    # Test sync streaming
    messages = [HumanMessage(content="Hello")]
    chunks = list(model.stream(messages))

    # Should have multiple chunks
    assert len(chunks) > 1

    # All should be AIMessageChunk
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)

    # Should reassemble to original
    complete = "".join(chunk.content for chunk in chunks)
    assert complete == "Hello"
