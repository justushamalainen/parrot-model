"""
Parrot Model - A local, mock LLM provider for development and testing.

The Parrot Model is a local LLM provider that "parrots" back responses in a predictable way,
supporting all standard LLM features including tool calls, streaming, and async operations.
It works completely offline, provides deterministic responses, and enables fast iteration
and testing of LLM-powered applications without requiring API keys or internet connectivity.

Key Features:
    - Works completely offline (no API keys needed)
    - Deterministic, controllable responses for reliable testing
    - Supports tool calls with custom encoding syntax
    - Streaming support (sync and async)
    - Integration with Pydantic AI and LangChain/LangGraph
    - Zero cost for development and testing

Example:
    >>> from parrot_model import ParrotModel, ParrotConfig
    >>> model = ParrotModel()
    >>> response = model.generate("Hello, world!")
    >>> print(response)
    "Hello, world!"

    >>> # With configuration
    >>> config = ParrotConfig(max_tokens=5)
    >>> model = ParrotModel(config=config)
    >>> response = model.generate("This is a long message that will be limited")
    >>> print(response)
    "This is a long message"

For more information, see the documentation at:
https://github.com/parrot-model/parrot-model
"""

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ParrotModel",
    "ParrotConfig",
]
