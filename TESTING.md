# Parrot Model Test Suite

This document describes the comprehensive test suite for the Parrot Model, including test coverage, how to run tests, and dependencies.

## Test Suite Overview

The test suite contains **exactly 15 tests** covering all critical functionality:

### Test Files

1. **tests/test_core.py** (11 tests) - Core functionality, tool calls, and streaming
2. **tests/test_adapters.py** (4 tests) - Framework adapters (Pydantic AI and LangChain)
3. **tests/conftest.py** - Shared fixtures and configuration

## Complete Test List (15 Tests)

### Core Functionality Tests (5 tests)

1. **test_basic_echo** - Validates basic echo mode behavior
2. **test_token_based_truncation** - Tests token-based text limiting
3. **test_character_based_truncation** - Tests character-based text limiting
4. **test_async_generation** - Validates async message generation
5. **test_configuration_handling** - Tests configuration initialization and storage

### Tool Call Tests (3 tests)

6. **test_single_tool_call_parsing** - Parses single tool call from message
7. **test_multiple_tool_calls** - Parses multiple tool calls from one message
8. **test_tool_calls_disabled** - Validates tool calls can be disabled

### Streaming Tests (3 tests)

9. **test_sync_streaming** - Tests synchronous streaming with chunks
10. **test_async_streaming** - Tests asynchronous streaming with chunks
11. **test_streaming_with_text_limits** - Streaming with truncation applied

### Integration Tests (4 tests)

12. **test_pydantic_ai_adapter_basic** - Pydantic AI adapter initialization and basic usage
13. **test_pydantic_ai_adapter_with_tool_calls** - Pydantic AI with tool call conversion
14. **test_langchain_adapter_basic** - LangChain adapter with message handling and tool calls
15. **test_langchain_adapter_streaming** - LangChain adapter streaming functionality

## Running the Tests

### Install Dependencies

```bash
# Install the package and core test dependencies
pip install -e .
pip install pytest pytest-asyncio

# Optional: Install adapter dependencies for full test coverage
pip install pydantic-ai  # For Pydantic AI adapter tests
pip install langchain-core  # For LangChain adapter tests
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=parrot_model --cov-report=html

# Run quietly (just show summary)
pytest tests/ -q
```

### Run Specific Test Files

```bash
# Run only core tests (11 tests)
pytest tests/test_core.py -v

# Run only adapter tests (4 tests)
pytest tests/test_adapters.py -v
```

### Run Specific Tests

```bash
# Run a single test by name
pytest tests/test_core.py::test_basic_echo -v

# Run all tool call tests
pytest tests/test_core.py -k "tool_call" -v

# Run all streaming tests
pytest tests/ -k "streaming" -v
```

## Test Dependencies

### Required Dependencies

- **pytest** - Test framework
- **pytest-asyncio** - For async test support
- **parrot-model** - The package being tested

### Optional Dependencies (for full coverage)

- **pydantic-ai** - Enables Pydantic AI adapter tests (2 tests)
  - Without: Tests are skipped with "pydantic-ai not installed"
- **langchain-core** - Enables LangChain adapter tests (2 tests)
  - Without: Tests are skipped with "langchain-core not installed"

### Test Results Without Optional Dependencies

```
13 passed, 2 skipped
```

### Test Results With All Dependencies

```
15 passed
```

## Test Fixtures

Defined in `tests/conftest.py`:

- **basic_model** - ParrotModel with default configuration
- **configured_model** - ParrotModel with custom config (truncation, delays, etc.)
- **model_without_tool_calls** - ParrotModel with tool calls disabled
- **sample_messages** - Dictionary of test messages for various scenarios

## Files Removed/Consolidated

The following test files were removed and consolidated into the new test suite:

- ✗ `/home/user/parrot-model/test_tool_calls.py` (standalone test script)
- ✗ `/home/user/parrot-model/tests/test_pydantic_ai_adapter.py` (old adapter tests)

All their functionality has been preserved and integrated into the new consolidated suite.

## Test Coverage

The 15 tests provide comprehensive coverage of:

- ✅ Core echo functionality
- ✅ Text truncation (token and character based)
- ✅ Synchronous and asynchronous generation
- ✅ Configuration handling
- ✅ Tool call parsing and encoding
- ✅ Tool call enable/disable
- ✅ Synchronous and asynchronous streaming
- ✅ Streaming with text limits
- ✅ Pydantic AI adapter integration
- ✅ LangChain adapter integration
- ✅ Adapter message conversion
- ✅ Adapter tool call handling
- ✅ Adapter streaming support

## Continuous Integration

To run these tests in CI/CD:

```bash
# Install with test dependencies
pip install -e ".[test]"

# Or install individually
pip install pytest pytest-asyncio

# Run tests with JUnit XML output for CI
pytest tests/ --junitxml=test-results.xml

# Run with coverage for reporting
pytest tests/ --cov=parrot_model --cov-report=xml
```

## Adding New Tests

When adding new functionality:

1. Keep the total at **15 tests maximum**
2. If adding a test, consider which existing test can be merged or removed
3. Add new fixtures to `conftest.py` if shared across multiple tests
4. Use descriptive test names: `test_<feature>_<scenario>`
5. Add docstrings explaining what each test validates
6. Mark optional dependency tests with `@pytest.mark.skipif`

## Test Design Principles

- **Focused**: Each test validates one specific behavior
- **Fast**: Tests run in < 1 second total
- **Independent**: Tests don't depend on each other
- **Conditional**: Adapter tests skip gracefully if dependencies missing
- **Clear**: Test names and docstrings clearly describe what's being tested
