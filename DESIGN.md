# Parrot Model Design Document

## Overview

The Parrot Model is a local, mock LLM provider designed for development and testing of LLM-powered applications without requiring access to actual LLM APIs. It "parrots" back responses in a predictable way, supporting all standard LLM features including tool calls, streaming, and async operations.

## Motivation

**Problem**: Developing and testing LLM-powered applications requires:
- API keys and internet connectivity
- Costs for API usage during development
- Non-deterministic responses making testing difficult
- Rate limits and quotas

**Solution**: A local "parrot" model that:
- Works completely offline
- Provides deterministic, controllable responses
- Supports all standard LLM interfaces
- Enables fast iteration and testing
- Zero cost for development

## Core Principles

1. **Interface Compatibility**: Support standard LLM interfaces (Pydantic AI, LangChain/LangGraph)
2. **Predictability**: Deterministic responses for reliable testing
3. **Simplicity**: Minimal configuration, easy to use
4. **Feature Completeness**: Support all LLM features (tool calls, streaming, etc.)
5. **Extensibility**: Easy to customize response behavior

## Architecture

### 1. Core Model Layer

The base `ParrotModel` class that implements core functionality:

```
parrot_model/
├── core/
│   ├── __init__.py
│   ├── base.py           # Base ParrotModel implementation
│   ├── config.py         # Configuration options
│   ├── response.py       # Response generation logic
│   └── tools.py          # Tool call handling
```

**Key Features**:
- **Echo Mode**: Returns input messages as output
- **Tool Call Support**: Parses and encodes tool parameters
- **Text Limiting**: Configurable max response length
- **Response Templates**: Customizable response patterns
- **Streaming Support**: Simulates streaming responses

### 2. Provider Adapters

Adapters for different LLM frameworks:

```
parrot_model/
├── adapters/
│   ├── __init__.py
│   ├── pydantic_ai.py    # Pydantic AI Provider implementation
│   ├── langchain.py      # LangChain BaseChatModel implementation
│   └── base.py           # Base adapter interface
```

### 3. Response Modes

Different modes of operation:

#### Echo Mode (Default)
Returns the user's message content back

```python
User: "Hello, how are you?"
Model: "Hello, how are you?"
```

#### Template Mode
Uses predefined templates with placeholders

```python
Template: "You said: {message}"
User: "Hello"
Model: "You said: Hello"
```

#### Tool Call Mode
Generates tool calls with encoded parameters

```python
User: "Get weather for London"
# Message includes special encoding: [TOOL:get_weather|city=London]
Model: Returns structured tool call for get_weather(city="London")
```

#### Streaming Mode
Returns responses character-by-character or token-by-token

## Feature Specifications

### 1. Message Encoding

**Tool Call Encoding**:
Users can encode tool calls directly in their messages using a special syntax:

```
Syntax: [TOOL:tool_name|param1=value1|param2=value2]

Examples:
"[TOOL:get_weather|city=London|units=metric]"
"[TOOL:search_database|query=users|limit=10]"
```

**Multiple Tool Calls**:
```
"[TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris]"
```

### 2. Text Limiting

Configure maximum response length:

```python
config = ParrotConfig(max_tokens=100)  # Limit to 100 tokens
config = ParrotConfig(max_chars=500)   # Limit to 500 characters
```

Truncation behavior:
- Word boundary truncation (default)
- Character truncation
- Token-based truncation (with simple tokenizer)

### 3. Response Configuration

```python
class ParrotConfig:
    mode: str = "echo"  # echo, template, smart
    max_tokens: int | None = None
    max_chars: int | None = None
    truncate_at_word: bool = True
    stream_delay_ms: int = 50  # Delay between chunks in streaming
    template: str | None = None
    enable_tool_calls: bool = True
    tool_call_pattern: str = r"\[TOOL:(\w+)\|(.+?)\]"
```

## Integration Specifications

### 1. Pydantic AI Integration

Implementation approach based on Pydantic AI 2025 architecture:

```python
from pydantic_ai import Model, Provider
from parrot_model import ParrotProvider

# Create provider
provider = ParrotProvider(config=ParrotConfig(max_tokens=100))

# Use with Pydantic AI
agent = Agent(model=provider)
result = agent.run_sync("Hello, world!")
```

**Implementation Requirements**:
- Subclass `Model` base class from `pydantic_ai.models`
- Implement `Provider` interface
- Support async/sync operations
- Handle streaming responses via `StreamedResponse`

### 2. LangChain/LangGraph Integration

Implementation based on `BaseChatModel`:

```python
from langchain.chat_models import BaseChatModel
from parrot_model import ParrotChatModel

# Create model
model = ParrotChatModel(max_tokens=100)

# Use with LangChain
from langchain.schema import HumanMessage
response = model.invoke([HumanMessage(content="Hello")])

# Use with LangGraph
from langgraph.graph import StateGraph
graph = StateGraph(state_schema)
graph.add_node("chat", lambda state: model.invoke(state["messages"]))
```

**Implementation Requirements**:
- Subclass `BaseChatModel` from `langchain_core.language_models.chat_models`
- Implement required methods:
  - `_generate()`: Synchronous generation
  - `_agenerate()`: Async generation
  - `_stream()`: Streaming (optional)
  - `_astream()`: Async streaming (optional)
- Support LangChain message types (HumanMessage, AIMessage, SystemMessage, etc.)
- Handle tool calls via function calling interface

## Tool Call Handling

### Design

Tool calls are the most complex feature. The model needs to:

1. **Parse tool call encoding** from messages
2. **Convert to framework-specific format** (Pydantic AI vs LangChain)
3. **Generate proper tool call responses**

### Tool Call Flow

```
User Message: "Get weather for London [TOOL:get_weather|city=London]"
    ↓
Parse tool encoding
    ↓
Extract: tool_name="get_weather", params={"city": "London"}
    ↓
Generate framework-specific tool call object
    ↓
Return as part of model response
```

### Framework-Specific Formats

**Pydantic AI**:
```python
# Tool call in response
ModelResponse(
    content="",
    tool_calls=[
        ToolCall(
            name="get_weather",
            args={"city": "London"}
        )
    ]
)
```

**LangChain**:
```python
# Tool call in AIMessage
AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}'
                }
            }
        ]
    }
)
```

## Project Structure

```
parrot-model/
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # User documentation
├── DESIGN.md                # This document
├── .gitignore              # Git ignore file
├── src/
│   └── parrot_model/
│       ├── __init__.py      # Public API
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py      # Core ParrotModel
│       │   ├── config.py    # Configuration classes
│       │   ├── response.py  # Response generation
│       │   ├── tools.py     # Tool call parsing/handling
│       │   └── streaming.py # Streaming support
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py      # Base adapter interface
│       │   ├── pydantic_ai.py
│       │   └── langchain.py
│       └── utils/
│           ├── __init__.py
│           ├── tokenizer.py # Simple tokenizer
│           └── parser.py    # Message parsing utilities
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_pydantic_ai.py
│   ├── test_langchain.py
│   └── test_tools.py
└── examples/
    ├── basic_usage.py
    ├── pydantic_ai_example.py
    ├── langchain_example.py
    └── tool_calls_example.py
```

## API Design

### Core API

```python
from parrot_model import ParrotModel, ParrotConfig

# Basic usage
model = ParrotModel()
response = model.generate("Hello, world!")
# Returns: "Hello, world!"

# With configuration
config = ParrotConfig(
    mode="echo",
    max_tokens=50,
    enable_tool_calls=True
)
model = ParrotModel(config=config)

# Streaming
for chunk in model.stream("Hello"):
    print(chunk, end="", flush=True)

# Async
async def main():
    response = await model.agenerate("Hello")
```

### Pydantic AI Adapter

```python
from parrot_model.adapters import ParrotPydanticProvider

# Create provider
provider = ParrotPydanticProvider(max_tokens=100)

# Use with Pydantic AI
from pydantic_ai import Agent

agent = Agent(model=provider)
result = agent.run_sync("Hello")
```

### LangChain Adapter

```python
from parrot_model.adapters import ParrotChatModel

# Create chat model
chat = ParrotChatModel(max_tokens=100)

# Use with LangChain
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Hello")]
response = chat.invoke(messages)

# Use in chains
from langchain_core.runnables import RunnablePassthrough
chain = RunnablePassthrough() | chat
```

## Configuration Options

### ParrotConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `str` | `"echo"` | Response mode: echo, template, smart |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in response |
| `max_chars` | `int \| None` | `None` | Maximum characters in response |
| `truncate_at_word` | `bool` | `True` | Truncate at word boundaries |
| `stream_delay_ms` | `int` | `50` | Delay between streaming chunks (ms) |
| `template` | `str \| None` | `None` | Template string for responses |
| `enable_tool_calls` | `bool` | `True` | Parse and handle tool calls |
| `tool_call_pattern` | `str` | See regex | Regex pattern for tool calls |
| `echo_system_messages` | `bool` | `False` | Include system messages in echo |
| `add_metadata` | `bool` | `False` | Add metadata to responses |

## Implementation Phases

### Phase 1: Core Foundation
- ✅ Design document
- [ ] Project setup (pyproject.toml, structure)
- [ ] Core ParrotModel implementation
- [ ] Basic configuration system
- [ ] Simple echo mode
- [ ] Text limiting functionality
- [ ] Unit tests for core

### Phase 2: Tool Call Support
- [ ] Tool call parsing from messages
- [ ] Tool parameter encoding/decoding
- [ ] Tool call response generation
- [ ] Tests for tool functionality

### Phase 3: Streaming Support
- [ ] Sync streaming implementation
- [ ] Async streaming implementation
- [ ] Configurable streaming delays
- [ ] Streaming tests

### Phase 4: Pydantic AI Integration
- [ ] Research Pydantic AI Provider interface
- [ ] Implement ParrotPydanticProvider
- [ ] Handle Pydantic AI message types
- [ ] Tool call integration for Pydantic AI
- [ ] Integration tests
- [ ] Example code

### Phase 5: LangChain Integration
- [ ] Implement BaseChatModel subclass
- [ ] Handle LangChain message types
- [ ] Tool call integration for LangChain
- [ ] Streaming support
- [ ] Integration tests
- [ ] Example code

### Phase 6: Documentation & Release
- [ ] README with examples
- [ ] API documentation
- [ ] Usage guides
- [ ] PyPI package preparation
- [ ] CI/CD setup
- [ ] Release v0.1.0

## Testing Strategy

### Unit Tests
- Core response generation
- Text limiting/truncation
- Tool call parsing
- Configuration handling
- Streaming functionality

### Integration Tests
- Pydantic AI integration
- LangChain integration
- Tool call end-to-end
- Streaming end-to-end

### Example-Based Tests
- Run example scripts as tests
- Verify expected outputs
- Test against both frameworks

## Dependencies

### Core Dependencies
```toml
[project]
dependencies = [
    "typing-extensions>=4.0.0",  # For type hints
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
pydantic-ai = [
    "pydantic-ai>=0.1.0",
]
langchain = [
    "langchain-core>=0.3.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.13.0",
]
all = [
    "parrot-model[pydantic-ai,langchain,dev]",
]
```

## Success Criteria

The project is successful when:

1. ✅ **Pydantic AI Integration**: Can be used as a drop-in model provider
2. ✅ **LangChain Integration**: Can be used as a ChatModel implementation
3. ✅ **Tool Calls**: Fully supports tool call encoding and execution
4. ✅ **Text Limiting**: Properly limits response length
5. ✅ **Streaming**: Supports both sync and async streaming
6. ✅ **Testing**: Enables deterministic testing of LLM applications
7. ✅ **Documentation**: Clear examples for both frameworks
8. ✅ **Zero Dependencies**: Core works without external LLM dependencies

## Future Enhancements

### V2 Features
- **Custom Response Scripts**: Python scripts to generate responses
- **Response Scenarios**: Pre-defined conversation scenarios
- **Latency Simulation**: Configurable delays to simulate API latency
- **Error Simulation**: Simulate API errors and rate limits
- **Token Usage Tracking**: Mock token counting for cost estimation
- **Multi-turn Conversations**: Stateful conversation handling
- **Response Validation**: Validate responses match expected schemas

### V3 Features
- **Web UI**: Interactive testing interface
- **Recording Mode**: Record real LLM responses for replay
- **Fuzzing Support**: Generate random valid responses
- **OpenAI API Server**: HTTP server with OpenAI-compatible API
- **Response Caching**: Cache responses for repeated queries

## Open Questions

1. **Token Counting**: How should we count tokens? Use a simple whitespace split or integrate tiktoken?
   - **Decision**: Start with simple split, add tiktoken as optional dependency later

2. **Tool Call ID Generation**: How to generate tool call IDs?
   - **Decision**: Use UUID4 for consistency across frameworks

3. **System Message Handling**: Should system messages be echoed by default?
   - **Decision**: No by default, make it configurable

4. **Error Handling**: Should we simulate LLM API errors?
   - **Decision**: Phase 1: No errors. Phase 2: Add configurable error simulation

5. **Message History**: Should the model maintain conversation history?
   - **Decision**: No - keep it stateless. Users can implement history at app level

## References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [LangChain Custom Chat Models](https://python.langchain.com/docs/how_to/custom_chat_model/)
- [LangGraph Documentation](https://www.langchain.com/langgraph)
- OpenAI API Format (for compatibility reference)

## Changelog

- **2025-11-08**: Initial design document created
