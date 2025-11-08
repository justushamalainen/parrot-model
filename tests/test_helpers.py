"""
Tests for the Pydantic AI and LangChain helper functions.

These tests verify that the helper functions correctly create tool calls
and handle dependencies appropriately.
"""

import json
import uuid

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
        from parrot_model.adapters.pydantic_ai_helpers import (
            encode_tool_call_in_prompt as pa_encode,
        )
        from parrot_model.adapters.langchain_helpers import (
            encode_tool_call_in_prompt as lc_encode,
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

    def test_create_tool_calls_multiple(self):
        """Test creating multiple tool calls at once."""
        pytest.importorskip("pydantic_ai", reason="Pydantic AI not installed")

        from parrot_model.adapters.pydantic_ai_helpers import create_tool_calls

        tool_calls = create_tool_calls(
            ("get_weather", {"city": "London"}),
            ("get_time", {"timezone": "UTC"}),
        )

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

    def test_create_tool_call_dict(self):
        """Test creating a tool call dictionary."""
        from parrot_model.adapters.langchain_helpers import create_tool_call_dict

        tool_dict = create_tool_call_dict("get_weather", city="London", units="metric")

        # Verify structure
        assert "id" in tool_dict
        assert "type" in tool_dict
        assert "function" in tool_dict
        assert tool_dict["type"] == "function"
        assert tool_dict["function"]["name"] == "get_weather"

        # Verify arguments are JSON-encoded
        args = json.loads(tool_dict["function"]["arguments"])
        assert args == {"city": "London", "units": "metric"}

    def test_create_tool_call_dict_with_custom_id(self):
        """Test creating a tool call dict with custom ID."""
        from parrot_model.adapters.langchain_helpers import create_tool_call_dict

        custom_id = "my-tool-call-123"
        tool_dict = create_tool_call_dict("test_tool", tool_call_id=custom_id)
        assert tool_dict["id"] == custom_id

    def test_create_tool_call_message_requires_langchain(self):
        """Test that create_tool_call_message checks for langchain."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from parrot_model.adapters.langchain_helpers import create_tool_call_message

        # Should work if langchain is installed
        message = create_tool_call_message("get_weather", city="London")

        # Verify it's an AIMessage
        assert hasattr(message, "content")
        assert hasattr(message, "additional_kwargs")
        assert "tool_calls" in message.additional_kwargs
        assert len(message.additional_kwargs["tool_calls"]) == 1

    def test_create_tool_call_message_with_content(self):
        """Test creating a tool call message with content."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from parrot_model.adapters.langchain_helpers import create_tool_call_message

        content = "Let me check that for you"
        message = create_tool_call_message(
            "get_weather", content=content, city="London"
        )

        assert message.content == content
        tool_call = message.additional_kwargs["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"

    def test_create_multi_tool_call_message(self):
        """Test creating a message with multiple tool calls."""
        pytest.importorskip("langchain_core", reason="LangChain not installed")

        from parrot_model.adapters.langchain_helpers import (
            create_multi_tool_call_message,
        )

        message = create_multi_tool_call_message(
            ("get_weather", {"city": "London"}),
            ("get_time", {"timezone": "UTC"}),
            ("calculate", {"x": 5, "y": 3}),
            content="Processing multiple operations",
        )

        assert message.content == "Processing multiple operations"
        tool_calls = message.additional_kwargs["tool_calls"]
        assert len(tool_calls) == 3
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"
        assert tool_calls[2]["function"]["name"] == "calculate"


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
