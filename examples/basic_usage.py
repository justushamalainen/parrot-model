"""
Basic usage examples for the Parrot Model.

This script demonstrates the fundamental features of ParrotModel,
including basic echo generation, text limiting, and async operations.
"""

import asyncio

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig


def example_basic_echo():
    """Basic example: Creating a ParrotModel and generating responses."""
    print("=" * 60)
    print("Example 1: Basic Echo Generation")
    print("=" * 60)

    # Create a ParrotModel with default configuration
    model = ParrotModel()

    # Generate a simple response
    message = "Hello, Parrot Model!"
    response = model.generate(message)

    print(f"Input:  {message}")
    print(f"Output: {response}")
    print()


def example_token_limiting():
    """Example demonstrating token-based text limiting."""
    print("=" * 60)
    print("Example 2: Token Limiting")
    print("=" * 60)

    # Create a model with token limit
    config = ParrotConfig(max_tokens=5)
    model = ParrotModel(config=config)

    # Generate a response - will be limited to 5 tokens
    message = "This is a very long message that will be truncated to only five tokens"
    response = model.generate(message)

    print(f"Input:  {message}")
    print(f"Config: max_tokens={config.max_tokens}")
    print(f"Output: {response}")
    print(f"Output tokens: {len(response.split())}")
    print()


def example_char_limiting():
    """Example demonstrating character-based text limiting."""
    print("=" * 60)
    print("Example 3: Character Limiting")
    print("=" * 60)

    # Create a model with character limit
    config = ParrotConfig(max_chars=30)
    model = ParrotModel(config=config)

    # Generate a response - will be limited to 30 characters
    message = "This message has way more than thirty characters in it!"
    response = model.generate(message)

    print(f"Input:  {message}")
    print(f"Config: max_chars={config.max_chars}")
    print(f"Output: {response}")
    print(f"Output chars: {len(response)}")
    print()


def example_word_truncation():
    """Example showing word-boundary truncation."""
    print("=" * 60)
    print("Example 4: Word-Boundary Truncation")
    print("=" * 60)

    # Compare truncation with and without word boundaries
    message = "Hello wonderful world"

    # Without word-boundary truncation (default)
    config1 = ParrotConfig(max_chars=15, truncate_at_word=False)
    model1 = ParrotModel(config=config1)
    response1 = model1.generate(message)

    # With word-boundary truncation
    config2 = ParrotConfig(max_chars=15, truncate_at_word=True)
    model2 = ParrotModel(config=config2)
    response2 = model2.generate(message)

    print(f"Input: {message}")
    print(f"\nWith truncate_at_word=False (cuts mid-word):")
    print(f"  Output: '{response1}'")
    print(f"\nWith truncate_at_word=True (cuts at word boundary):")
    print(f"  Output: '{response2}'")
    print()


async def example_async_generation():
    """Example demonstrating async generation."""
    print("=" * 60)
    print("Example 5: Async Generation")
    print("=" * 60)

    # Create a model
    model = ParrotModel()

    # Generate response asynchronously
    message = "Async generation is useful for non-blocking operations"
    response = await model.agenerate(message)

    print(f"Input:  {message}")
    print(f"Output: {response}")
    print()


async def example_multiple_async():
    """Example running multiple async generations concurrently."""
    print("=" * 60)
    print("Example 6: Concurrent Async Generations")
    print("=" * 60)

    # Create a model
    model = ParrotModel()

    # Define multiple messages
    messages = [
        "First concurrent message",
        "Second concurrent message",
        "Third concurrent message"
    ]

    # Generate all responses concurrently
    print("Generating 3 responses concurrently...")
    responses = await asyncio.gather(
        *[model.agenerate(msg) for msg in messages]
    )

    # Display results
    for i, (msg, resp) in enumerate(zip(messages, responses), 1):
        print(f"\n{i}. Input:  {msg}")
        print(f"   Output: {resp}")
    print()


def example_configuration_options():
    """Example showing various configuration options."""
    print("=" * 60)
    print("Example 7: Configuration Options")
    print("=" * 60)

    # Create a custom configuration
    config = ParrotConfig(
        mode="echo",                  # Response mode (currently only 'echo')
        max_tokens=20,                # Limit by tokens
        max_chars=None,               # No character limit
        truncate_at_word=True,        # Truncate at word boundaries
        stream_delay_ms=50,           # Delay between stream chunks (ms)
        enable_tool_calls=True,       # Enable tool call parsing
        echo_system_messages=False    # Don't echo system messages
    )

    model = ParrotModel(config=config)

    message = "This message demonstrates custom configuration with truncation"
    response = model.generate(message)

    print("Configuration:")
    print(f"  max_tokens: {config.max_tokens}")
    print(f"  truncate_at_word: {config.truncate_at_word}")
    print(f"  enable_tool_calls: {config.enable_tool_calls}")
    print(f"\nInput:  {message}")
    print(f"Output: {response}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Parrot Model - Basic Usage" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Run synchronous examples
    example_basic_echo()
    example_token_limiting()
    example_char_limiting()
    example_word_truncation()
    example_configuration_options()

    # Run async examples
    asyncio.run(example_async_generation())
    asyncio.run(example_multiple_async())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
