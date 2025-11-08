"""
Tests for the Pydantic AI and LangChain helper functions.

These tests verify that the helper functions correctly create tool calls
and handle dependencies appropriately.
"""


import pytest


class TestSharedUtilities:
    """Tests for shared utility functions used by both helper modules."""

    def test_encode_tool_call_in_prompt(self):
        """Test encoding tool calls in prompt format."""
        from parrot_model.adapters._tool_call_utils import encode_tool_call_in_prompt

        # Test with parameters
        result = encode_tool_call_in_prompt("get_weather", city="London", units="metric")
        assert result == "[TOOL:get_weather|city=London|units=metric]"

        # Test without parameters
        result = encode_tool_call_in_prompt("simple_tool")
        assert result == "[TOOL:simple_tool|]"

        # Test parameter ordering (should be sorted alphabetically)
        result = encode_tool_call_in_prompt("test", z="last", a="first", m="middle")
        assert result == "[TOOL:test|a=first|m=middle|z=last]"

    def test_generate_tool_call_id_with_none(self):
        """Test ID generation when no ID is provided."""
        from parrot_model.adapters._tool_call_utils import generate_tool_call_id

        # Generate two IDs and ensure they're different
        id1 = generate_tool_call_id(None)
        id2 = generate_tool_call_id(None)

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Each call should generate unique ID
        assert len(id1) == 36  # UUID4 format (with hyphens)

    def test_generate_tool_call_id_with_custom(self):
        """Test ID generation when custom ID is provided."""
        from parrot_model.adapters._tool_call_utils import generate_tool_call_id

        custom_id = "my-custom-id-123"
        result = generate_tool_call_id(custom_id)

        assert result == custom_id  # Should return the provided ID

    def test_both_helpers_use_same_shared_function(self):
        """Test that both helper modules use the same shared encoding function."""
        from parrot_model.adapters.langchain_helpers import (
            encode_tool_call_in_prompt as lc_encode,
        )
        from parrot_model.adapters.pydantic_ai_helpers import (
            encode_tool_call_in_prompt as pa_encode,
        )

        # Both should produce the same result
        params = {"city": "London", "units": "metric"}
        pa_result = pa_encode("get_weather", **params)
        lc_result = lc_encode("get_weather", **params)

        assert pa_result == lc_result
        assert pa_result == "[TOOL:get_weather|city=London|units=metric]"


class TestPydanticAIHelpers:
    """Tests for Pydantic AI helper functions."""

    def test_encode_tool_call_in_prompt(self):
        """Test encoding tool calls in prompt format."""
        from parrot_model.adapters.pydantic_ai_helpers import (
            encode_tool_call_in_prompt,
        )

        # Test with parameters
        result = encode_tool_call_in_prompt("get_weather", city="London", units="metric")
        assert result == "[TOOL:get_weather|city=London|units=metric]"

        # Test without parameters
        result = encode_tool_call_in_prompt("simple_tool")
        assert result == "[TOOL:simple_tool|]"

        # Test parameter ordering (should be sorted)
        result = encode_tool_call_in_prompt("test", z="last", a="first", m="middle")
        assert result == "[TOOL:test|a=first|m=middle|z=last]"

    def test_create_tool_call_requires_pydantic_ai(self):
        """Test that create_tool_call checks for pydantic-ai."""
        pytest.importorskip("pydantic_ai", reason="Pydantic AI not installed")

        from parrot_model.adapters.pydantic_ai_helpers import create_tool_call

        # Should work if pydantic_ai is installed
        tool_call = create_tool_call("get_weather", city="London")
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"city": "London"}
        assert isinstance(tool_call.tool_call_id, str)

    def test_create_tool_call_with_custom_id(self):
        """Test creating a tool call with a custom ID."""
        pytest.importorskip("pydantic_ai", reason="Pydantic AI not installed")

        from parrot_model.adapters.pydantic_ai_helpers import create_tool_call

        custom_id = "my-custom-id-123"
        tool_call = create_tool_call("test_tool", tool_call_id=custom_id, param="value")
        assert tool_call.tool_call_id == custom_id

    def test_create_multiple_tool_calls_with_list_comprehension(self):
        """Test creating multiple tool calls with list comprehension."""
        pytest.importorskip("pydantic_ai", reason="Pydantic AI not installed")

        from parrot_model.adapters.pydantic_ai_helpers import create_tool_call

        # Users can create multiple tool calls with list comprehension
        calls = [
            ("get_weather", {"city": "London"}),
            ("get_time", {"timezone": "UTC"}),
        ]
        tool_calls = [create_tool_call(name, **params) for name, params in calls]

        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "get_weather"
        assert tool_calls[1].tool_name == "get_time"
        assert tool_calls[0].args == {"city": "London"}
        assert tool_calls[1].args == {"timezone": "UTC"}


class TestLangChainHelpers:
    """Tests for LangChain helper functions."""

    def test_encode_tool_call_in_prompt(self):
        """Test encoding tool calls in prompt format."""
        from parrot_model.adapters.langchain_helpers import encode_tool_call_in_prompt

        # Test with parameters
        result = encode_tool_call_in_prompt("get_weather", city="London", units="metric")
        assert result == "[TOOL:get_weather|city=London|units=metric]"

        # Test without parameters
        result = encode_tool_call_in_prompt("simple_tool")
        assert result == "[TOOL:simple_tool|]"

    def test_create_tool_call_requires_langchain(self):
        """Test that create_tool_call checks for langchain."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from parrot_model.adapters.langchain_helpers import create_tool_call

        # Should work if langchain is installed
        tool_call = create_tool_call("get_weather", city="London")

        assert tool_call.name == "get_weather"
        assert tool_call.args == {"city": "London"}
        assert isinstance(tool_call.id, str)

    def test_create_tool_call_with_custom_id(self):
        """Test creating a tool call with custom ID."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from parrot_model.adapters.langchain_helpers import create_tool_call

        custom_id = "my-tool-call-123"
        tool_call = create_tool_call("test_tool", tool_call_id=custom_id, param="value")
        assert tool_call.id == custom_id
        assert tool_call.name == "test_tool"
        assert tool_call.args == {"param": "value"}

    def test_create_multiple_tool_calls_and_message(self):
        """Test creating multiple tool calls and using them in an AIMessage."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from langchain_core.messages import AIMessage

        from parrot_model.adapters.langchain_helpers import create_tool_call

        # Create multiple tool calls with list comprehension
        tool_calls = [
            create_tool_call("get_weather", city="London"),
            create_tool_call("get_time", timezone="UTC"),
            create_tool_call("calculate", x=5, y=3)
        ]

        # Use them in an AIMessage
        message = AIMessage(
            content="Processing multiple operations",
            tool_calls=tool_calls
        )

        assert message.content == "Processing multiple operations"
        assert len(message.tool_calls) == 3
        assert message.tool_calls[0].name == "get_weather"
        assert message.tool_calls[1].name == "get_time"
        assert message.tool_calls[2].name == "calculate"


class TestHelperConsistency:
    """Tests to ensure consistency between Pydantic AI and LangChain helpers."""

    def test_encoding_consistency(self):
        """Test that both helpers produce the same encoding."""
        from parrot_model.adapters.langchain_helpers import (
            encode_tool_call_in_prompt as lc_encode,
        )
        from parrot_model.adapters.pydantic_ai_helpers import (
            encode_tool_call_in_prompt as pa_encode,
        )

        # Test same parameters produce same encoding
        params = {"city": "London", "units": "metric", "lang": "en"}
        lc_result = lc_encode("get_weather", **params)
        pa_result = pa_encode("get_weather", **params)

        assert lc_result == pa_result

        # Test no parameters
        assert lc_encode("simple") == pa_encode("simple")
