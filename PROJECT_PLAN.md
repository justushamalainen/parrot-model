# Parrot Model - Project Implementation Plan

## Executive Summary

**Project**: Parrot Model - A local, mock LLM provider for development and testing

**Goal**: Create a library that mimics LLM behavior locally without API calls, supporting both Pydantic AI and LangChain/LangGraph frameworks.

**Status**: Planning Complete ✅

## Key Design Decisions

### 1. Architecture: Layered Approach

```
┌─────────────────────────────────────────┐
│   User Applications (Pydantic AI,      │
│   LangChain, LangGraph)                 │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│   Adapter Layer                         │
│   - ParrotPydanticProvider              │
│   - ParrotChatModel (LangChain)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│   Core Layer                            │
│   - ParrotModel (base implementation)   │
│   - Response generation                 │
│   - Tool call handling                  │
│   - Streaming support                   │
└─────────────────────────────────────────┘
```

**Rationale**:
- Separation of concerns: core logic vs. framework-specific adapters
- Easy to add new framework support
- Core can be tested independently

### 2. Tool Call Encoding: Custom Syntax

**Decision**: Use `[TOOL:name|param=value]` syntax in messages

**Example**: `"Get weather [TOOL:get_weather|city=London|units=metric]"`

**Rationale**:
- Simple and readable
- Easy to parse with regex
- Doesn't require JSON knowledge
- Can be mixed with natural language

**Alternative Considered**: JSON encoding in messages
- Rejected because it's verbose and less readable

### 3. Response Modes: Start Simple, Expand Later

**Phase 1**: Echo mode only
**Phase 2**: Add template mode
**Phase 3**: Add smart mode (context-aware responses)

**Rationale**:
- YAGNI principle - start with minimum viable product
- Echo mode covers 80% of testing use cases
- Can iterate based on user feedback

### 4. Text Limiting: Multiple Strategies

Support both token-based and character-based limiting:
- `max_tokens`: Simple whitespace-based tokenization
- `max_chars`: Character count
- `truncate_at_word`: Intelligent word boundary truncation

**Rationale**:
- Different use cases need different limiting strategies
- Avoid heavy dependencies (no tiktoken initially)
- Users can choose what makes sense for their tests

### 5. Framework Support Priority

**Priority 1**: Pydantic AI (newer, growing adoption)
**Priority 2**: LangChain/LangGraph (established, widely used)

**Rationale**:
- Pydantic AI is simpler to implement (newer, cleaner API)
- LangChain has more complex message types and legacy
- Both are important for adoption

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Core functionality working

**Tasks**:
1. Project setup (pyproject.toml, structure)
2. Core `ParrotModel` class
3. Basic echo mode
4. Configuration system (`ParrotConfig`)
5. Text limiting (tokens + chars)
6. Unit tests

**Deliverable**: Working core that can echo messages with length limits

### Phase 2: Tool Calls (Week 2)
**Goal**: Tool calling support

**Tasks**:
1. Tool call parser (regex-based)
2. Parameter extraction and validation
3. Tool call response generation
4. Integration with core model
5. Comprehensive tests

**Deliverable**: Messages with `[TOOL:...]` syntax generate proper tool calls

### Phase 3: Streaming (Week 2)
**Goal**: Streaming support

**Tasks**:
1. Sync streaming (`stream()`)
2. Async streaming (`astream()`)
3. Configurable delays
4. Chunk size configuration
5. Tests for streaming

**Deliverable**: Both sync and async streaming working

### Phase 4: Pydantic AI Integration (Week 3)
**Goal**: Works as Pydantic AI provider

**Tasks**:
1. Research Pydantic AI Model/Provider interface
2. Implement `ParrotPydanticProvider`
3. Message type conversions
4. Tool call format conversion
5. Integration tests
6. Example scripts

**Deliverable**: Can be used with `pydantic_ai.Agent`

### Phase 5: LangChain Integration (Week 3-4)
**Goal**: Works as LangChain ChatModel

**Tasks**:
1. Implement `BaseChatModel` subclass
2. Implement required methods (_generate, _agenerate)
3. Implement streaming methods (_stream, _astream)
4. LangChain message type handling
5. Tool call format for LangChain
6. Integration tests
7. Example scripts

**Deliverable**: Can be used with LangChain chains and LangGraph

### Phase 6: Polish & Release (Week 4)
**Goal**: Production-ready release

**Tasks**:
1. Complete documentation
2. Usage examples
3. CI/CD setup (GitHub Actions)
4. PyPI package preparation
5. CHANGELOG
6. Contributing guidelines
7. Release v0.1.0

**Deliverable**: Published package on PyPI

## Technical Specifications

### Core Classes

```python
# Core configuration
@dataclass
class ParrotConfig:
    mode: str = "echo"
    max_tokens: int | None = None
    max_chars: int | None = None
    truncate_at_word: bool = True
    stream_delay_ms: int = 50
    enable_tool_calls: bool = True
    # ... more options

# Core model
class ParrotModel:
    def __init__(self, config: ParrotConfig | None = None):
        ...

    def generate(self, message: str) -> str:
        """Synchronous generation"""

    async def agenerate(self, message: str) -> str:
        """Async generation"""

    def stream(self, message: str) -> Iterator[str]:
        """Sync streaming"""

    async def astream(self, message: str) -> AsyncIterator[str]:
        """Async streaming"""

# Tool call parsing
class ToolCallParser:
    def parse(self, message: str) -> list[ToolCall]:
        """Extract tool calls from message"""

    @staticmethod
    def encode(tool_name: str, **params) -> str:
        """Create tool call encoding"""
```

### Adapter Interfaces

```python
# Pydantic AI
class ParrotPydanticProvider(Model):
    """Pydantic AI Model implementation"""

    async def request(
        self,
        messages: list[Message],
        ...
    ) -> ModelResponse:
        ...

# LangChain
class ParrotChatModel(BaseChatModel):
    """LangChain BaseChatModel implementation"""

    def _generate(
        self,
        messages: list[BaseMessage],
        ...
    ) -> ChatResult:
        ...

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        ...
    ) -> ChatResult:
        ...
```

## Directory Structure

```
parrot-model/
├── pyproject.toml           # Build system, dependencies
├── README.md                # User documentation
├── DESIGN.md                # Design document (detailed)
├── PROJECT_PLAN.md          # This file
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore
├── .github/
│   └── workflows/
│       ├── test.yml        # CI tests
│       └── publish.yml     # PyPI publishing
├── src/
│   └── parrot_model/
│       ├── __init__.py      # Public API exports
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py      # ParrotModel class
│       │   ├── config.py    # ParrotConfig
│       │   ├── response.py  # Response generation
│       │   ├── tools.py     # Tool call handling
│       │   └── streaming.py # Streaming utilities
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py      # Base adapter
│       │   ├── pydantic_ai.py
│       │   └── langchain.py
│       └── utils/
│           ├── __init__.py
│           ├── tokenizer.py # Simple tokenizer
│           └── parser.py    # Parsing utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_core.py        # Core tests
│   ├── test_config.py      # Config tests
│   ├── test_tools.py       # Tool call tests
│   ├── test_streaming.py   # Streaming tests
│   ├── test_pydantic_ai.py # Pydantic AI tests
│   └── test_langchain.py   # LangChain tests
└── examples/
    ├── basic_usage.py
    ├── configuration.py
    ├── tool_calls.py
    ├── streaming.py
    ├── pydantic_ai_example.py
    ├── langchain_example.py
    └── langgraph_example.py
```

## Dependencies Strategy

### Core (Minimal)
- No external dependencies for core functionality
- Python 3.10+ (for modern type hints)

### Optional Dependencies
```toml
[project.optional-dependencies]
pydantic-ai = ["pydantic-ai>=0.1.0"]
langchain = ["langchain-core>=0.3.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.13.0",
]
all = ["parrot-model[pydantic-ai,langchain,dev]"]
```

**Rationale**:
- Keep core lightweight
- Users only install what they need
- Easy to add new integrations

## Testing Strategy

### 1. Unit Tests (pytest)
- Test each component in isolation
- Mock external dependencies
- 90%+ code coverage target

### 2. Integration Tests
- Test with actual Pydantic AI Agent
- Test with actual LangChain chains
- Test with LangGraph graphs

### 3. Example-Based Tests
- Run all example scripts as tests
- Verify expected outputs
- Catch breaking changes in examples

### 4. Type Checking (mypy)
- Strict mode
- Ensure type hints are accurate

### 5. Linting (ruff)
- Consistent code style
- Catch common errors

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Pydantic AI API changes | Medium | High | Pin version, monitor releases |
| LangChain API changes | Low | Medium | Well-established API |
| Complex tool call formats | Medium | Medium | Start simple, iterate |
| Streaming implementation complexity | Low | Medium | Use async patterns |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Low adoption | Medium | High | Good docs, examples |
| Competing solutions | Low | Medium | Focus on simplicity |
| Maintenance burden | Low | Low | Keep scope limited |

## Success Metrics

### Phase 1 Success
- [ ] Core model echoes messages correctly
- [ ] Text limiting works for both tokens and chars
- [ ] 90%+ test coverage
- [ ] All unit tests passing

### Phase 2 Success
- [ ] Tool calls parse correctly
- [ ] Multiple tool calls supported
- [ ] Framework-agnostic tool representation

### Phase 3 Success
- [ ] Streaming works sync and async
- [ ] Configurable delays work
- [ ] Memory efficient (no buffering)

### Phase 4 Success
- [ ] Works with Pydantic AI Agent
- [ ] Tool calls work in Pydantic AI
- [ ] Examples run successfully

### Phase 5 Success
- [ ] Works with LangChain chains
- [ ] Works with LangGraph graphs
- [ ] Tool calls work in LangChain
- [ ] Examples run successfully

### Final Release Success
- [ ] Published on PyPI
- [ ] Documentation complete
- [ ] At least 3 complete examples per framework
- [ ] CI/CD pipeline working

## Next Steps

1. **Setup project structure**
   ```bash
   # Create pyproject.toml
   # Create directory structure
   # Setup .gitignore
   # Initialize git
   ```

2. **Implement Phase 1**
   - Start with `ParrotConfig`
   - Implement basic `ParrotModel`
   - Add text limiting
   - Write tests

3. **Iterate through phases**
   - Complete each phase before moving to next
   - Write tests as you go
   - Create examples for each feature

4. **Release preparation**
   - Documentation review
   - Example testing
   - PyPI setup
   - Announcement

## Open Questions & Decisions Needed

### 1. Python Version Support
**Question**: Support Python 3.10+ or 3.8+?
**Recommendation**: 3.10+ for modern type hints
**Decision**: 3.10+

### 2. Tokenization Method
**Question**: Simple split or integrate tiktoken?
**Recommendation**: Start with simple split, add tiktoken as optional
**Decision**: Simple split for v0.1.0

### 3. Error Simulation
**Question**: Should we simulate API errors?
**Recommendation**: Not in v0.1.0, add in v0.2.0
**Decision**: Phase 1: No errors

### 4. Response Caching
**Question**: Should repeated queries return cached responses?
**Recommendation**: Not needed for deterministic echo mode
**Decision**: No caching in v0.1.0

### 5. Package Name
**Question**: `parrot-model` or `parrot-llm`?
**Recommendation**: `parrot-model` (mirrors "language model")
**Decision**: `parrot-model`

## Resources

### Documentation
- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [LangChain Custom Chat Models](https://python.langchain.com/docs/how_to/custom_chat_model/)
- [LangGraph Docs](https://www.langchain.com/langgraph)

### Similar Projects (for inspiration)
- Mock servers (WireMock, etc.)
- Test doubles patterns
- OpenAI API mocking libraries

### Tools
- pytest for testing
- ruff for linting
- mypy for type checking
- GitHub Actions for CI/CD

## Timeline Estimate

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Foundation | 3 days | Day 1 | Day 3 |
| Phase 2: Tool Calls | 2 days | Day 4 | Day 5 |
| Phase 3: Streaming | 2 days | Day 6 | Day 7 |
| Phase 4: Pydantic AI | 3 days | Day 8 | Day 10 |
| Phase 5: LangChain | 4 days | Day 11 | Day 14 |
| Phase 6: Release | 2 days | Day 15 | Day 16 |
| **Total** | **~3 weeks** | | |

**Note**: This is an aggressive timeline for a solo developer. More realistic might be 4-6 weeks.

## Conclusion

The Parrot Model project has a clear architecture, well-defined phases, and achievable goals. The layered approach with adapters makes it extensible, and the focus on testing ensures reliability.

The key to success is:
1. **Start simple** (echo mode)
2. **Test thoroughly** (90%+ coverage)
3. **Iterate based on feedback** (don't over-engineer)
4. **Document well** (examples for every feature)

Ready to begin implementation! 🦜
