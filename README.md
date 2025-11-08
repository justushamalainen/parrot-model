# 🦜 Parrot Model

A local, deterministic LLM provider for development and testing - because sometimes you just need your LLM to repeat what you say.

[![PyPI version](https://badge.fury.io/py/parrot-model.svg)](https://badge.fury.io/py/parrot-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is Parrot Model?

Parrot Model is a mock LLM provider that lets you develop and test LLM-powered applications **without requiring API keys, internet connectivity, or spending money on API calls**. It "parrots" back responses in predictable ways, making it perfect for:

- 🧪 **Testing**: Write deterministic tests for your LLM applications
- 🚀 **Development**: Iterate quickly without API costs or rate limits
- 📚 **Learning**: Understand LLM integration patterns without external dependencies
- 🔧 **Debugging**: Predictable responses make debugging easier

## Features

- ✅ **Zero Dependencies**: Works completely offline, no API keys needed
- ✅ **Framework Support**: Works with Pydantic AI and LangChain/LangGraph
- ✅ **Tool Calls**: Full support for function/tool calling
- ✅ **Streaming**: Supports both sync and async streaming
- ✅ **Deterministic**: Same input = same output, every time
- ✅ **Configurable**: Control response length, templates, and behavior
- ✅ **Easy to Use**: Drop-in replacement for real LLM providers

## Quick Start

### Installation

```bash
pip install parrot-model
```

With optional dependencies:

```bash
# For Pydantic AI
pip install parrot-model[pydantic-ai]

# For LangChain
pip install parrot-model[langchain]

# For everything
pip install parrot-model[all]
```

### Basic Usage

```python
from parrot_model import ParrotModel

model = ParrotModel()
response = model.generate("Hello, world!")
print(response)  # Output: "Hello, world!"
```

### With Pydantic AI

```python
from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
from parrot_model.core.config import ParrotConfig
from pydantic_ai import Agent

# Use parrot model as your LLM provider
model = ParrotPydanticModel(config=ParrotConfig(max_tokens=100))
agent = Agent(model)

result = agent.run_sync("What's the weather like?")
print(result.data)
```

### With LangChain

```python
from parrot_model.adapters import ParrotChatModel
from langchain_core.messages import HumanMessage

chat = ParrotChatModel(max_tokens=100)
messages = [HumanMessage(content="Hello!")]
response = chat.invoke(messages)
print(response.content)
```

### With LangGraph

```python
from parrot_model.adapters import ParrotChatModel
from langgraph.graph import StateGraph, MessagesState

chat = ParrotChatModel()

def chatbot(state: MessagesState):
    return {"messages": [chat.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
# ... configure your graph
```

## Tool Calls

Parrot Model supports tool calls through a special encoding syntax:

```python
from parrot_model import ParrotModel

model = ParrotModel(enable_tool_calls=True)

# Encode a tool call in your message
message = "Get the weather for London [TOOL:get_weather|city=London|units=metric]"
response = model.generate(message)

# The model will parse and return a proper tool call
```

**Tool Call Syntax:**
```
[TOOL:function_name|param1=value1|param2=value2]
```

**Examples:**
```python
# Single tool call
"[TOOL:get_weather|city=London]"

# Multiple parameters
"[TOOL:search_database|query=users|limit=10|offset=0]"

# Multiple tool calls in one message
"Check weather in [TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris]"
```

### Tool Call Helpers

Parrot Model provides helper functions to easily create tool calls for both Pydantic AI and LangChain/LangGraph:

#### Pydantic AI Helpers

```python
from parrot_model.adapters.pydantic_ai_helpers import (
    create_tool_call,
    create_tool_calls,
    encode_tool_call_in_prompt
)

# Create a single tool call
tool_call = create_tool_call("get_weather", city="London", units="metric")

# Create multiple tool calls at once
tool_calls = create_tool_calls(
    ("get_weather", {"city": "London"}),
    ("get_time", {"timezone": "UTC"})
)

# Encode a tool call for use in prompts
encoding = encode_tool_call_in_prompt("get_weather", city="London")
# Returns: "[TOOL:get_weather|city=London]"
```

#### LangChain/LangGraph Helpers

```python
from parrot_model.adapters.langchain_helpers import (
    create_tool_call_message,
    create_multi_tool_call_message,
    encode_tool_call_in_prompt
)

# Create an AI message with a tool call
message = create_tool_call_message("get_weather", city="London")

# Create a message with multiple tool calls
message = create_multi_tool_call_message(
    ("get_weather", {"city": "London"}),
    ("get_time", {"timezone": "UTC"}),
    content="Checking multiple locations"
)

# Encode a tool call for use in prompts
encoding = encode_tool_call_in_prompt("get_weather", city="London")
# Returns: "[TOOL:get_weather|city=London]"
```

## Configuration

```python
from parrot_model import ParrotModel, ParrotConfig

config = ParrotConfig(
    mode="echo",              # Response mode: echo, template, smart
    max_tokens=100,           # Limit response length
    max_chars=500,            # Alternative: limit by characters
    truncate_at_word=True,    # Truncate at word boundaries
    stream_delay_ms=50,       # Delay between streaming chunks
    enable_tool_calls=True,   # Enable tool call parsing
)

model = ParrotModel(config=config)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `str` | `"echo"` | Response mode (echo/template/smart) |
| `max_tokens` | `int` | `None` | Maximum tokens in response |
| `max_chars` | `int` | `None` | Maximum characters in response |
| `truncate_at_word` | `bool` | `True` | Truncate at word boundaries |
| `stream_delay_ms` | `int` | `50` | Delay between chunks (ms) |
| `template` | `str` | `None` | Template for responses |
| `enable_tool_calls` | `bool` | `True` | Parse tool call syntax |

## Response Modes

### Echo Mode (Default)

Simply returns what you send:

```python
model = ParrotModel(mode="echo")
response = model.generate("Hello!")
# Returns: "Hello!"
```

### Template Mode

Use a template with placeholders:

```python
config = ParrotConfig(
    mode="template",
    template="You said: {message}"
)
model = ParrotModel(config=config)
response = model.generate("Hello!")
# Returns: "You said: Hello!"
```

## Streaming

Both sync and async streaming are supported:

```python
# Sync streaming
for chunk in model.stream("Hello, world!"):
    print(chunk, end="", flush=True)

# Async streaming
async def main():
    async for chunk in model.astream("Hello, world!"):
        print(chunk, end="", flush=True)
```

## Use Cases

### Testing

```python
import pytest
from parrot_model.adapters import ParrotChatModel

@pytest.fixture
def chat_model():
    return ParrotChatModel()

def test_chat_response(chat_model):
    """Test with deterministic responses"""
    from langchain_core.messages import HumanMessage

    response = chat_model.invoke([
        HumanMessage(content="test message")
    ])

    assert response.content == "test message"
    assert response.response_metadata is not None
```

### Development

```python
from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
from pydantic_ai import Agent

# Develop your agent logic without API costs
model = ParrotPydanticModel()
agent = Agent(model)

# Test your prompts and flows locally
result = agent.run_sync("Test prompt")
```

### Learning

```python
# Understand how LLM integrations work
# without needing real API credentials

from parrot_model.adapters import ParrotChatModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chat = ParrotChatModel()
prompt = PromptTemplate.from_template("Hello {name}!")
chain = prompt | chat

# See exactly what's happening without API calls
response = chain.invoke({"name": "World"})
```

## Examples

Check out the [examples/](./examples/) directory for more:

- `basic_usage.py` - Core functionality
- `pydantic_ai_example.py` - Pydantic AI integration
- `pydantic_ai_tool_helpers.py` - Pydantic AI tool call helpers
- `langchain_example.py` - LangChain integration
- `langchain_tool_helpers.py` - LangChain/LangGraph tool call helpers
- `tool_calls_example.py` - Tool calling examples

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/parrot-model.git
cd parrot-model

# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Run linting
ruff check .

# Type checking
mypy src/parrot_model
```

## Why "Parrot"?

Because like a parrot, it repeats what you say - but unlike a real parrot, it does exactly what you tell it to do, when you tell it to do it. Perfect for testing!

## Comparison with Real LLMs

| Feature | Real LLM | Parrot Model |
|---------|----------|--------------|
| API Key Required | ✅ Yes | ❌ No |
| Internet Required | ✅ Yes | ❌ No |
| Costs Money | ✅ Yes | ❌ No |
| Deterministic | ❌ No | ✅ Yes |
| Good for Testing | ❌ Limited | ✅ Excellent |
| Good for Production | ✅ Yes | ❌ No |
| Streaming Support | ✅ Yes | ✅ Yes |
| Tool Calls | ✅ Yes | ✅ Yes |

## Roadmap

- [x] Core echo functionality
- [x] Tool call support
- [x] Pydantic AI integration
- [x] LangChain integration
- [ ] Response templates
- [ ] Custom response scripts
- [ ] Latency simulation
- [ ] Error simulation
- [ ] OpenAI-compatible API server
- [ ] Web UI for testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the need for better local LLM testing tools
- Built to work with [Pydantic AI](https://ai.pydantic.dev/) and [LangChain](https://python.langchain.com/)

## Support

- 📖 [Documentation](./DESIGN.md)
- 🐛 [Issue Tracker](https://github.com/yourusername/parrot-model/issues)
- 💬 [Discussions](https://github.com/yourusername/parrot-model/discussions)

---

Made with ❤️ for developers who want to test LLM applications without the hassle.
