# Parrot Model Examples

This directory contains comprehensive example scripts demonstrating how to use the Parrot Model with various features and LLM frameworks.

## Overview

The Parrot Model is a local LLM simulation tool designed for testing, development, and learning. These examples show you how to use its features effectively.

## Quick Start

Install the Parrot Model with all optional dependencies:

```bash
pip install parrot-model[all]
```

Or install only what you need:

```bash
# Core functionality only
pip install parrot-model

# For Pydantic AI examples
pip install parrot-model[pydantic-ai]

# For LangChain/LangGraph examples
pip install parrot-model[langchain]
```

## Examples

### 1. Basic Usage (`basic_usage.py`)

**What it demonstrates:**
- Creating a ParrotModel instance
- Basic echo generation
- Text limiting (by tokens and characters)
- Word-boundary truncation
- Async generation
- Concurrent async operations
- Configuration options

**How to run:**
```bash
python examples/basic_usage.py
```

**Dependencies:** None (core only)

**Expected output:**
- Demonstrates how the model echoes back input messages
- Shows how token and character limits truncate output
- Displays the difference between word-boundary and character truncation
- Shows async generation capabilities

**Key concepts:**
```python
from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

# Basic usage
model = ParrotModel()
response = model.generate("Hello!")

# With configuration
config = ParrotConfig(max_tokens=10, truncate_at_word=True)
model = ParrotModel(config=config)

# Async usage
response = await model.agenerate("Async message")
```

---

### 2. Tool Calls (`tool_calls_example.py`)

**What it demonstrates:**
- Encoding tool calls in messages using `[TOOL:name|param=value]` syntax
- Parsing tool calls from responses
- Multiple tool calls in a single message
- Retrieving parsed tool calls
- Programmatic tool call encoding with `ToolCallParser.encode()`
- Enabling/disabling tool call parsing
- Typical tool call workflows

**How to run:**
```bash
python examples/tool_calls_example.py
```

**Dependencies:** None (core only)

**Expected output:**
- Shows how to embed tool calls in messages
- Displays parsed tool call objects with names, parameters, and IDs
- Demonstrates programmatic encoding and parsing
- Shows workflow for handling tool calls

**Key concepts:**
```python
from parrot_model.core.base import ParrotModel
from parrot_model.core.tools import ToolCallParser

# Basic tool call
model = ParrotModel()
response = model.generate("Get [TOOL:get_weather|city=London]")
tool_calls = model.get_tool_calls()

# Programmatic encoding
encoded = ToolCallParser.encode("search", query="test", limit="10")
# Returns: "[TOOL:search|limit=10|query=test]"
```

---

### 3. Streaming (`streaming_example.py`)

**What it demonstrates:**
- Synchronous streaming (character by character)
- Asynchronous streaming
- Custom chunk sizes
- Custom streaming delays
- Concurrent async streams
- Streaming with text truncation
- Progress tracking during streaming

**How to run:**
```bash
python examples/streaming_example.py
```

**Dependencies:** None (core only)

**Expected output:**
- Shows real-time streaming of responses
- Demonstrates different chunk sizes and their effects
- Shows timing differences with various delay configurations
- Displays concurrent streaming operations

**Key concepts:**
```python
from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig

# Sync streaming
model = ParrotModel()
for chunk in model.stream("Hello", chunk_size=1):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in model.astream("Hello", chunk_size=5):
    print(chunk, end="", flush=True)

# Custom delay
config = ParrotConfig(stream_delay_ms=50)
model = ParrotModel(config=config)
```

---

### 4. Pydantic AI Integration (`pydantic_ai_integration.py`)

**What it demonstrates:**
- Using ParrotPydanticModel with Pydantic AI agents
- Basic agent usage
- Structured output with Pydantic models
- Tool calls with Pydantic AI
- Streaming with Pydantic AI
- Configuration options

**How to run:**
```bash
python examples/pydantic_ai_integration.py
```

**Dependencies:** `pydantic-ai`
```bash
pip install parrot-model[pydantic-ai]
```

**Expected output:**
- Shows Parrot Model working as a Pydantic AI model
- Demonstrates agent creation and execution
- Shows structured output parsing
- Displays tool call integration
- Shows streaming responses

**Key concepts:**
```python
from pydantic_ai import Agent
from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
from parrot_model.core.config import ParrotConfig

# Basic agent
model = ParrotPydanticModel()
agent = Agent(model)
result = await agent.run("Hello!")

# With tools
@agent.tool
def get_weather(ctx, city: str) -> str:
    return f"Weather in {city}: Sunny"

# Streaming
async with agent.run_stream("Hello!") as result:
    async for message in result.stream_text():
        print(message, end="", flush=True)
```

---

### 5. LangChain Integration (`langchain_integration.py`)

**What it demonstrates:**
- Using ParrotChatModel as a LangChain chat model
- Working with LangChain messages (HumanMessage, SystemMessage, etc.)
- LCEL (LangChain Expression Language) chains
- Tool calls in LangChain format
- Streaming (sync and async)
- Batch processing
- Simple LangGraph workflow
- Output parsing

**How to run:**
```bash
python examples/langchain_integration.py
```

**Dependencies:** `langchain-core` and optionally `langgraph`
```bash
pip install parrot-model[langchain]
```

**Expected output:**
- Shows Parrot Model working as a LangChain chat model
- Demonstrates message handling and conversation flow
- Shows LCEL chain composition
- Displays tool call integration in OpenAI format
- Shows LangGraph workflow example

**Key concepts:**
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from parrot_model.adapters.langchain import ParrotChatModel

# Basic usage
model = ParrotChatModel()
response = model.invoke([HumanMessage(content="Hello!")])

# LCEL chain
prompt = ChatPromptTemplate.from_template("Process: {text}")
chain = prompt | model | StrOutputParser()
result = chain.invoke({"text": "data"})

# Streaming
for chunk in model.stream([HumanMessage(content="Hi")]):
    print(chunk.content, end="", flush=True)
```

---

## Running Examples

### Run Individual Examples

```bash
# Run specific examples
python examples/basic_usage.py
python examples/tool_calls_example.py
python examples/streaming_example.py
python examples/pydantic_ai_integration.py
python examples/langchain_integration.py
```

### Run All Examples

```bash
# Run all examples in sequence (requires all dependencies)
for script in examples/*.py; do
    [ "$script" != "examples/README.md" ] && python "$script"
done
```

## Example Output Format

All examples follow a consistent format:
- Clear section headers with `=` separators
- Descriptive titles for each demonstration
- Input and output clearly labeled
- Configuration settings displayed when relevant
- Informative comments explaining key concepts

## Common Patterns

### Configuration

```python
from parrot_model.core.config import ParrotConfig

config = ParrotConfig(
    mode="echo",              # Response mode
    max_tokens=100,           # Limit by tokens
    max_chars=None,           # Limit by characters
    truncate_at_word=True,    # Truncate at word boundaries
    stream_delay_ms=50,       # Streaming delay (milliseconds)
    enable_tool_calls=True,   # Enable tool call parsing
    echo_system_messages=False # Don't echo system messages
)
```

### Error Handling

Examples include graceful handling of missing dependencies:

```python
try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    print("Pydantic AI not installed. Some examples will be skipped.")
```

## Learning Path

We recommend exploring the examples in this order:

1. **basic_usage.py** - Start here to understand core concepts
2. **streaming_example.py** - Learn about streaming responses
3. **tool_calls_example.py** - Understand tool call functionality
4. **pydantic_ai_integration.py** - See framework integration (Pydantic AI)
5. **langchain_integration.py** - See framework integration (LangChain/LangGraph)

## Troubleshooting

### Import Errors

If you get import errors, ensure you've installed the required dependencies:

```bash
# For Pydantic AI examples
pip install pydantic-ai

# For LangChain examples
pip install langchain-core

# For LangGraph examples
pip install langgraph

# Or install everything
pip install parrot-model[all]
```

### Module Not Found

Make sure the package is installed:

```bash
pip install -e .  # If running from the repository
# OR
pip install parrot-model  # If installed from PyPI
```

## Additional Resources

- **Main README**: See `/README.md` for project overview
- **Design Document**: See `/DESIGN.md` for architecture details
- **API Documentation**: Check docstrings in source files
- **Tests**: See `/tests/` directory for more usage examples

## Contributing Examples

If you create useful examples, consider contributing them! Please ensure:
- Examples are self-contained and runnable
- Code includes clear comments
- Dependencies are handled gracefully
- Output is informative and well-formatted
- Examples follow the existing style

## License

These examples are part of the Parrot Model project and share the same license.
