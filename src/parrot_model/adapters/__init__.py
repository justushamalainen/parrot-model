"""
Framework adapters for the Parrot Model.

This module contains adapters that integrate the Parrot Model with
various LLM frameworks, including Pydantic AI and LangChain/LangGraph.
Each adapter translates between the framework-specific interfaces and
the core Parrot Model functionality.

It also includes helper modules for easily creating tool calls:
- pydantic_ai_helpers: Helpers for creating Pydantic AI tool calls
- langchain_helpers: Helpers for creating LangChain/LangGraph tool calls
"""

__all__ = []

# Pydantic AI adapter - exported conditionally based on availability
try:
    from parrot_model.adapters.pydantic_ai import ParrotPydanticModel

    __all__.append("ParrotPydanticModel")
except ImportError:
    # Pydantic AI not installed, don't export
    pass

# Pydantic AI helpers - exported conditionally based on availability
try:
    from parrot_model.adapters import pydantic_ai_helpers

    __all__.append("pydantic_ai_helpers")
except ImportError:
    # Pydantic AI not installed, don't export helpers
    pass

# LangChain adapter - exported conditionally based on availability
try:
    from parrot_model.adapters.langchain import ParrotChatModel

    __all__.append("ParrotChatModel")
except ImportError:
    # LangChain not installed, don't export
    pass

# LangChain helpers - exported conditionally based on availability
try:
    from parrot_model.adapters import langchain_helpers

    __all__.append("langchain_helpers")
except ImportError:
    # LangChain not installed, don't export helpers
    pass
