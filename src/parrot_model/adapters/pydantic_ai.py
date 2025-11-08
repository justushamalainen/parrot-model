"""
Pydantic AI adapter for the Parrot Model.

This module implements the Pydantic AI Model interface,
allowing the Parrot Model to be used as a drop-in replacement for
real LLM providers in Pydantic AI applications.

Note: This adapter targets Pydantic AI v0.1.x+ (2025 API).
      Make sure you have pydantic-ai installed: pip install pydantic-ai
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

if TYPE_CHECKING:
    from parrot_model.core.tools import ToolCall

# Try to import Pydantic AI components
try:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        ModelSettings,
        RequestUsage,
        StreamedResponse,
    )

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

    # Create stub types for type checking when pydantic_ai is not installed
    if TYPE_CHECKING:
        from pydantic_ai.messages import (
            ModelMessage,
            ModelResponse,
        )
        from pydantic_ai.models import (
            Model,
            ModelRequestParameters,
            ModelSettings,
            RequestUsage,
            StreamedResponse,
        )


def _check_pydantic_ai_available():
    """Check if pydantic-ai is installed and raise helpful error if not."""
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "Pydantic AI is not installed. "
            "Please install it with: pip install pydantic-ai\n"
            "Or install parrot-model with the pydantic-ai extra: "
            "pip install parrot-model[pydantic-ai]"
        )


class ParrotPydanticModel(Model if PYDANTIC_AI_AVAILABLE else object):
    """
    Pydantic AI Model adapter for ParrotModel.

    This adapter allows ParrotModel to be used as a standard Pydantic AI model,
    providing a local, deterministic alternative to cloud-based LLM providers
    for development and testing.

    The adapter converts Pydantic AI's message format to simple strings for
    ParrotModel, then converts the responses back to Pydantic AI's ModelResponse
    format with proper tool call handling.

    Attributes:
        parrot_model: Internal ParrotModel instance for generating responses.
        _model_name: Name identifier for this model instance.
        _system: Provider/system identifier (always "parrot" for this model).

    Example:
        >>> from pydantic_ai import Agent
        >>> from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
        >>>
        >>> # Create a Parrot model with configuration
        >>> model = ParrotPydanticModel(
        ...     model_name="parrot-echo",
        ...     config=ParrotConfig(max_tokens=100)
        ... )
        >>>
        >>> # Use with Pydantic AI Agent
        >>> agent = Agent(model)
        >>> result = agent.run_sync("Hello, world!")
        >>> print(result.data)
        'Hello, world!'

    Example with tools:
        >>> from pydantic_ai import Agent
        >>> from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
        >>>
        >>> # Create model with tool calls enabled
        >>> model = ParrotPydanticModel()
        >>>
        >>> # Define a tool
        >>> agent = Agent(model)
        >>> @agent.tool
        ... def get_weather(city: str) -> str:
        ...     return f"Weather in {city}: Sunny, 72°F"
        >>>
        >>> # User message can trigger tool calls using [TOOL:name|param=value] syntax
        >>> result = agent.run_sync("Check [TOOL:get_weather|city=London]")
    """

    def __init__(
        self,
        model_name: str = "parrot-echo",
        config: ParrotConfig | None = None,
        system_name: str = "parrot",
    ):
        """
        Initialize the ParrotPydanticModel.

        Args:
            model_name: Name identifier for this model. Defaults to "parrot-echo".
            config: Optional ParrotConfig for customizing behavior. If None, uses
                   default ParrotConfig with tool calls enabled.
            system_name: Provider/system identifier. Defaults to "parrot".

        Raises:
            ImportError: If pydantic-ai is not installed.

        Example:
            >>> from parrot_model import ParrotConfig
            >>> model = ParrotPydanticModel(
            ...     model_name="parrot-test",
            ...     config=ParrotConfig(max_tokens=50, stream_delay_ms=10)
            ... )
        """
        _check_pydantic_ai_available()

        # Initialize parent Model class if available
        if PYDANTIC_AI_AVAILABLE:
            super().__init__()

        self._model_name = model_name
        self._system = system_name

        # Create ParrotModel with config
        self.parrot_model = ParrotModel(config=config or ParrotConfig())

    @property
    def model_name(self) -> str:
        """
        Return the model name identifier.

        Returns:
            The model name string.
        """
        return self._model_name

    @property
    def system(self) -> str:
        """
        Return the system/provider identifier.

        Returns:
            The system name (always "parrot" for this implementation).
        """
        return self._system

    def _messages_to_string(self, messages: list[ModelMessage]) -> str:
        """
        Convert Pydantic AI messages to a single string for ParrotModel.

        Extracts text content from various message types and combines them
        into a single string. System messages are included if configured,
        and tool return messages are formatted appropriately.

        Args:
            messages: List of Pydantic AI ModelMessage objects.

        Returns:
            Combined string representation of all messages.

        Example:
            >>> # System message + user message
            >>> messages = [
            ...     ModelRequest(parts=[SystemPromptPart(content="You are helpful")]),
            ...     ModelRequest(parts=[UserPromptPart(content="Hello")])
            ... ]
            >>> model._messages_to_string(messages)
            'System: You are helpful\\nUser: Hello'
        """
        parts = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        # Include system messages if configured
                        if self.parrot_model.config.echo_system_messages:
                            parts.append(f"System: {part.content}")
                    elif isinstance(part, UserPromptPart):
                        parts.append(f"User: {part.content}")
                    elif isinstance(part, ToolReturnPart):
                        # Include tool results in context
                        parts.append(f"Tool {part.tool_name} returned: {part.content}")
            elif isinstance(message, ModelResponse):
                # Include previous model responses for context
                for part in message.parts:
                    if isinstance(part, TextPart):
                        parts.append(f"Assistant: {part.content}")

        # Join all parts with newlines
        combined = "\n".join(parts)

        # If no user messages found, just return empty string
        # ParrotModel will echo it back
        return combined

    def _parrot_tools_to_pydantic(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallPart]:
        """
        Convert ParrotModel ToolCall objects to Pydantic AI ToolCallPart objects.

        Args:
            tool_calls: List of ParrotModel ToolCall objects.

        Returns:
            List of Pydantic AI ToolCallPart objects.

        Example:
            >>> from parrot_model.core.tools import ToolCall
            >>> tool_call = ToolCall(
            ...     name="get_weather",
            ...     parameters={"city": "London"},
            ...     id="abc-123"
            ... )
            >>> parts = model._parrot_tools_to_pydantic([tool_call])
            >>> parts[0].tool_name
            'get_weather'
            >>> parts[0].args
            {'city': 'London'}
        """
        pydantic_parts = []

        for tool_call in tool_calls:
            pydantic_parts.append(
                ToolCallPart(
                    tool_name=tool_call.name,
                    args=tool_call.parameters,
                    tool_call_id=tool_call.id,
                )
            )

        return pydantic_parts

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a non-streaming request to the ParrotModel.

        Converts Pydantic AI messages to a string, generates a response using
        ParrotModel, and converts the result back to Pydantic AI's ModelResponse
        format with proper tool call handling.

        Args:
            messages: List of ModelMessage objects representing the conversation.
            model_settings: Optional model-specific settings (currently unused).
            model_request_parameters: Request parameters (currently unused).

        Returns:
            ModelResponse containing the generated text and/or tool calls.

        Example:
            >>> import asyncio
            >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
            >>>
            >>> model = ParrotPydanticModel()
            >>> messages = [
            ...     ModelRequest(parts=[UserPromptPart(content="Hello")])
            ... ]
            >>> response = asyncio.run(model.request(messages, None, {}))
            >>> response.parts[0].content
            'User: Hello'
        """
        # Convert messages to string
        prompt = self._messages_to_string(messages)

        # Generate response using ParrotModel (async)
        response_text = await self.parrot_model.agenerate(prompt)

        # Get tool calls if any were parsed
        tool_calls = self.parrot_model.get_tool_calls()

        # Build response parts
        parts: list[TextPart | ToolCallPart] = []

        # Add text part if there's any text content
        if response_text:
            parts.append(TextPart(content=response_text))

        # Add tool call parts
        if tool_calls:
            pydantic_tool_calls = self._parrot_tools_to_pydantic(tool_calls)
            parts.extend(pydantic_tool_calls)

        # Create ModelResponse with timestamp
        return ModelResponse(
            parts=parts,
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc),
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """
        Make a streaming request to the ParrotModel.

        Creates a streaming response that yields chunks of text as they are
        generated. Tool calls are included in the final response.

        Args:
            messages: List of ModelMessage objects representing the conversation.
            model_settings: Optional model-specific settings (currently unused).
            model_request_parameters: Request parameters (currently unused).

        Yields:
            StreamedResponse that can be iterated to get response chunks.

        Example:
            >>> import asyncio
            >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
            >>>
            >>> async def stream_example():
            ...     model = ParrotPydanticModel()
            ...     messages = [
            ...         ModelRequest(parts=[UserPromptPart(content="Hello")])
            ...     ]
            ...     async with model.request_stream(messages, None, {}) as response:
            ...         async for text in response.stream_text():
            ...             print(text, end="", flush=True)
            >>> asyncio.run(stream_example())
            User: Hello
        """
        # Convert messages to string
        prompt = self._messages_to_string(messages)

        # Create and yield a ParrotStreamedResponse
        streamed_response = ParrotStreamedResponse(
            parrot_model=self.parrot_model, prompt=prompt, model_name=self.model_name
        )

        yield streamed_response


if PYDANTIC_AI_AVAILABLE:

    class ParrotStreamedResponse(StreamedResponse):
        """
        Streamed response implementation for ParrotModel.

        This class wraps ParrotModel's async streaming to provide the
        StreamedResponse interface expected by Pydantic AI.

        Attributes:
            parrot_model: The ParrotModel instance to stream from.
            prompt: The prompt string to generate a response for.
            model_name: Name of the model for response metadata.
            _text_chunks: Accumulated text chunks.
            _finished: Whether streaming is complete.
        """

        def __init__(
            self, parrot_model: ParrotModel, prompt: str, model_name: str
        ) -> None:
            """
            Initialize the streamed response.

            Args:
                parrot_model: ParrotModel instance to stream from.
                prompt: Prompt to generate response for.
                model_name: Model name for metadata.
            """
            self.parrot_model = parrot_model
            self.prompt = prompt
            self.model_name = model_name
            self._text_chunks: list[str] = []
            self._finished = False

        async def stream_text(self, *, debounce_by: float | None = None) -> AsyncIterator[str]:
            """
            Stream text chunks from the model.

            Args:
                debounce_by: Optional debounce delay (currently unused).

            Yields:
                Text chunks as they are generated.

            Example:
                >>> async def example():
                ...     async for chunk in response.stream_text():
                ...         print(chunk, end="")
            """
            # Use ParrotModel's async streaming
            async for chunk in self.parrot_model.astream(self.prompt):
                self._text_chunks.append(chunk)
                yield chunk

            self._finished = True

        def get(self, *, final: bool = False) -> ModelResponse:
            """
            Get the current or final ModelResponse.

            Args:
                final: Whether to wait for streaming to complete.

            Returns:
                ModelResponse with accumulated text and any tool calls.

            Raises:
                RuntimeError: If called with final=True before streaming completes.

            Example:
                >>> response = streamed_response.get(final=True)
                >>> print(response.parts[0].content)
            """
            if final and not self._finished:
                raise RuntimeError(
                    "Cannot get final response before streaming is complete. "
                    "Make sure to iterate through stream_text() first."
                )

            # Combine all text chunks
            full_text = "".join(self._text_chunks)

            # Get tool calls
            tool_calls = self.parrot_model.get_tool_calls()

            # Build response parts
            parts: list[TextPart | ToolCallPart] = []

            if full_text:
                parts.append(TextPart(content=full_text))

            if tool_calls:
                # Convert ParrotModel tool calls to Pydantic AI format
                for tool_call in tool_calls:
                    parts.append(
                        ToolCallPart(
                            tool_name=tool_call.name,
                            args=tool_call.parameters,
                            tool_call_id=tool_call.id,
                        )
                    )

            return ModelResponse(
                parts=parts,
                model_name=self.model_name,
                timestamp=datetime.now(timezone.utc),
            )

        def usage(self) -> RequestUsage | None:
            """
            Get token usage information.

            ParrotModel doesn't track real token usage, so this returns None.

            Returns:
                None (token tracking not supported).

            Example:
                >>> usage = streamed_response.usage()
                >>> print(usage)  # None
            """
            # ParrotModel doesn't track token usage
            return None


# Export the main class
__all__ = ["ParrotPydanticModel"]
