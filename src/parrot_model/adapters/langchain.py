"""
LangChain adapter for the Parrot Model.

This module implements the LangChain BaseChatModel interface,
allowing the Parrot Model to be used in LangChain chains and
LangGraph workflows as a standard chat model.

Example:
    >>> from parrot_model.adapters.langchain import ParrotChatModel
    >>> from langchain_core.messages import HumanMessage
    >>>
    >>> # Basic usage
    >>> model = ParrotChatModel()
    >>> response = model.invoke([HumanMessage(content="Hello!")])
    >>> print(response.content)
    'Hello!'
    >>>
    >>> # With streaming
    >>> for chunk in model.stream([HumanMessage(content="Hello world")]):
    ...     print(chunk.content, end="", flush=True)
    Hello world
    >>>
    >>> # With tool calls
    >>> message = HumanMessage(content="Get [TOOL:weather|city=London]")
    >>> response = model.invoke([message])
    >>> response.additional_kwargs["tool_calls"]
    [{'id': '...', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{"city": "London"}'}}]
"""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

# Try to import LangChain dependencies
try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create placeholder types for type hints
    BaseChatModel = object  # type: ignore
    BaseMessage = object  # type: ignore
    ChatResult = object  # type: ignore
    ChatGenerationChunk = object  # type: ignore
    CallbackManagerForLLMRun = object  # type: ignore
    AsyncCallbackManagerForLLMRun = object  # type: ignore


class ParrotChatModel(BaseChatModel):  # type: ignore
    """
    LangChain chat model adapter for the Parrot Model.

    This class implements the LangChain BaseChatModel interface, making it compatible
    with LangChain chains, LCEL (LangChain Expression Language), and LangGraph workflows.
    It wraps the core ParrotModel and handles message conversion, streaming, and tool calls.

    Attributes:
        parrot_config: ParrotConfig instance controlling model behavior.

    Example:
        >>> from parrot_model.adapters.langchain import ParrotChatModel
        >>> from parrot_model.core.config import ParrotConfig
        >>> from langchain_core.messages import HumanMessage, SystemMessage
        >>>
        >>> # Create with custom config
        >>> config = ParrotConfig(max_tokens=50, stream_delay_ms=20)
        >>> model = ParrotChatModel(parrot_config=config)
        >>>
        >>> # Use with messages
        >>> messages = [
        ...     SystemMessage(content="You are a helpful assistant."),
        ...     HumanMessage(content="What is 2+2?")
        ... ]
        >>> response = model.invoke(messages)
        >>> print(response.content)
        'What is 2+2?'
        >>>
        >>> # Use in a chain (LCEL)
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> prompt = ChatPromptTemplate.from_messages([
        ...     ("system", "You are a helpful assistant."),
        ...     ("human", "{input}")
        ... ])
        >>> chain = prompt | model
        >>> response = chain.invoke({"input": "Hello!"})
        >>> print(response.content)
        'Hello!'
        >>>
        >>> # With streaming
        >>> async for chunk in model.astream(messages):
        ...     print(chunk.content, end="", flush=True)
        What is 2+2?
        >>>
        >>> # With tool calls
        >>> message = HumanMessage(content="Get [TOOL:search|query=Python]")
        >>> response = model.invoke([message])
        >>> print(response.additional_kwargs["tool_calls"])
        [{'id': '...', 'type': 'function', 'function': {'name': 'search', 'arguments': '{"query": "Python"}'}}]
    """

    # Pydantic model field for the configuration
    parrot_config: ParrotConfig = ParrotConfig()

    def __init__(
        self,
        parrot_config: Optional[ParrotConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize the ParrotChatModel.

        Args:
            parrot_config: Optional ParrotConfig instance. If None, uses default configuration.
            **kwargs: Additional keyword arguments passed to BaseChatModel.

        Raises:
            ImportError: If langchain_core is not installed.

        Example:
            >>> model = ParrotChatModel()  # Use defaults
            >>> model = ParrotChatModel(parrot_config=ParrotConfig(max_tokens=100))  # Custom config
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain-core"
            )

        # Set parrot_config in kwargs if provided
        if parrot_config is not None:
            kwargs["parrot_config"] = parrot_config

        # Call parent __init__ with kwargs
        super().__init__(**kwargs)

    @property
    def _model(self) -> ParrotModel:
        """
        Get or create the internal ParrotModel instance.

        Returns:
            ParrotModel instance.
        """
        # Create model lazily from parrot_config
        if not hasattr(self, "_cached_model"):
            self._cached_model = ParrotModel(self.parrot_config)
        return self._cached_model

    def _convert_messages_to_string(self, messages: list[BaseMessage]) -> str:
        """
        Convert a list of LangChain messages to a string for the ParrotModel.

        Combines all messages into a single string, with special handling for
        different message types. System messages are excluded by default unless
        echo_system_messages is True in the config.

        Args:
            messages: List of LangChain BaseMessage objects.

        Returns:
            Combined message string.

        Example:
            >>> from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            >>> model = ParrotChatModel()
            >>> messages = [
            ...     SystemMessage(content="You are helpful."),
            ...     HumanMessage(content="Hello!"),
            ...     AIMessage(content="Hi there!"),
            ...     HumanMessage(content="How are you?")
            ... ]
            >>> model._convert_messages_to_string(messages)
            'Hello! Hi there! How are you?'
        """
        parts = []
        for message in messages:
            # Skip system messages unless config says to include them
            if message.type == "system" and not self.parrot_config.echo_system_messages:
                continue

            # Extract content
            content = message.content
            if isinstance(content, list):
                # Handle multi-modal content (just use text parts for now)
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            parts.append(str(content))

        return " ".join(parts)

    def _create_ai_message(self, content: str, tool_calls: list) -> AIMessage:
        """
        Create an AIMessage with content and optional tool calls.

        Formats tool calls in OpenAI format and includes them in additional_kwargs.

        Args:
            content: The message content.
            tool_calls: List of ToolCall objects from ParrotModel.

        Returns:
            AIMessage with content and tool calls.

        Example:
            >>> model = ParrotChatModel()
            >>> tool_calls = [ToolCall(name="search", parameters={"q": "test"}, id="123")]
            >>> message = model._create_ai_message("Result", tool_calls)
            >>> message.content
            'Result'
            >>> message.additional_kwargs["tool_calls"][0]["function"]["name"]
            'search'
        """
        additional_kwargs: dict[str, Any] = {}

        # Format tool calls in OpenAI format if present
        if tool_calls:
            additional_kwargs["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.parameters),
                    },
                }
                for tc in tool_calls
            ]

        return AIMessage(content=content, additional_kwargs=additional_kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response synchronously.

        This is the main generation method called by LangChain's invoke() method.

        Args:
            messages: List of LangChain messages.
            stop: Optional list of stop sequences (currently not used).
            run_manager: Optional callback manager for LangChain callbacks.
            **kwargs: Additional generation parameters.

        Returns:
            ChatResult containing the generated message.

        Example:
            >>> model = ParrotChatModel()
            >>> from langchain_core.messages import HumanMessage
            >>> result = model._generate([HumanMessage(content="Hello!")])
            >>> result.generations[0].message.content
            'Hello!'
        """
        # Convert messages to string
        message_str = self._convert_messages_to_string(messages)

        # Generate response
        response = self._model.generate(message_str)

        # Get tool calls
        tool_calls = self._model.get_tool_calls()

        # Create AI message
        ai_message = self._create_ai_message(response, tool_calls)

        # Create generation
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response asynchronously.

        This is the async version of _generate, called by LangChain's ainvoke() method.

        Args:
            messages: List of LangChain messages.
            stop: Optional list of stop sequences (currently not used).
            run_manager: Optional async callback manager for LangChain callbacks.
            **kwargs: Additional generation parameters.

        Returns:
            ChatResult containing the generated message.

        Example:
            >>> import asyncio
            >>> model = ParrotChatModel()
            >>> from langchain_core.messages import HumanMessage
            >>> async def test():
            ...     result = await model._agenerate([HumanMessage(content="Hello!")])
            ...     return result.generations[0].message.content
            >>> asyncio.run(test())
            'Hello!'
        """
        # Convert messages to string
        message_str = self._convert_messages_to_string(messages)

        # Generate response asynchronously
        response = await self._model.agenerate(message_str)

        # Get tool calls
        tool_calls = self._model.get_tool_calls()

        # Create AI message
        ai_message = self._create_ai_message(response, tool_calls)

        # Create generation
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream a response synchronously.

        This method is called by LangChain's stream() method and yields chunks
        of the response as they are generated.

        Args:
            messages: List of LangChain messages.
            stop: Optional list of stop sequences (currently not used).
            run_manager: Optional callback manager for LangChain callbacks.
            **kwargs: Additional generation parameters.

        Yields:
            ChatGenerationChunk objects containing response chunks.

        Example:
            >>> model = ParrotChatModel()
            >>> from langchain_core.messages import HumanMessage
            >>> chunks = list(model._stream([HumanMessage(content="Hi")]))
            >>> "".join(chunk.message.content for chunk in chunks)
            'Hi'
        """
        # Convert messages to string
        message_str = self._convert_messages_to_string(messages)

        # Stream the response
        for chunk in self._model.stream(message_str):
            # Create chunk message
            chunk_message = AIMessageChunk(content=chunk)

            # Yield generation chunk
            yield ChatGenerationChunk(message=chunk_message)

        # Note: Tool calls are included in the final chunk in some implementations,
        # but for simplicity, we just stream the content. Tool calls can be retrieved
        # from _model.get_tool_calls() after streaming completes.

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Stream a response asynchronously.

        This is the async version of _stream, called by LangChain's astream() method.

        Args:
            messages: List of LangChain messages.
            stop: Optional list of stop sequences (currently not used).
            run_manager: Optional async callback manager for LangChain callbacks.
            **kwargs: Additional generation parameters.

        Yields:
            ChatGenerationChunk objects containing response chunks.

        Example:
            >>> import asyncio
            >>> model = ParrotChatModel()
            >>> from langchain_core.messages import HumanMessage
            >>> async def test():
            ...     chunks = []
            ...     async for chunk in model._astream([HumanMessage(content="Hi")]):
            ...         chunks.append(chunk.message.content)
            ...     return "".join(chunks)
            >>> asyncio.run(test())
            'Hi'
        """
        # Convert messages to string
        message_str = self._convert_messages_to_string(messages)

        # Stream the response asynchronously
        async for chunk in self._model.astream(message_str):
            # Create chunk message
            chunk_message = AIMessageChunk(content=chunk)

            # Yield generation chunk
            yield ChatGenerationChunk(message=chunk_message)

    @property
    def _llm_type(self) -> str:
        """
        Return the LLM type identifier.

        Returns:
            The string "parrot" identifying this as a Parrot model.

        Example:
            >>> model = ParrotChatModel()
            >>> model._llm_type
            'parrot'
        """
        return "parrot"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """
        Return identifying parameters for the model.

        These parameters are used by LangChain for caching and identification.

        Returns:
            Dictionary of configuration parameters.

        Example:
            >>> model = ParrotChatModel(ParrotConfig(max_tokens=100))
            >>> params = model._identifying_params
            >>> params["max_tokens"]
            100
        """
        return {
            "mode": self.parrot_config.mode,
            "max_tokens": self.parrot_config.max_tokens,
            "max_chars": self.parrot_config.max_chars,
            "truncate_at_word": self.parrot_config.truncate_at_word,
            "stream_delay_ms": self.parrot_config.stream_delay_ms,
            "enable_tool_calls": self.parrot_config.enable_tool_calls,
            "echo_system_messages": self.parrot_config.echo_system_messages,
        }
