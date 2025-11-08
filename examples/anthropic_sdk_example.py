"""
Anthropic SDK Adapter Example

This example demonstrates how to use the Parrot Model as a drop-in replacement
for the Anthropic Python SDK. The ParrotModel mimics the Anthropic client interface,
allowing you to develop and test Claude-based applications without API keys or costs.

Features demonstrated:
- Basic message creation
- Streaming responses
- Async operations
- Tool use
- System prompts
- Custom configuration
"""

import asyncio

from parrot_model import ParrotConfig
from parrot_model.adapters import Anthropic, AsyncAnthropic


def basic_example():
    """Basic synchronous message creation example."""
    print("=" * 60)
    print("Basic Example: Anthropic SDK Interface")
    print("=" * 60)

    # Create client - just like the real Anthropic SDK
    client = Anthropic()

    # Create a message
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Model name for compatibility
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude!"}],
    )

    print(f"Message ID: {message.id}")
    print(f"Model: {message.model}")
    print(f"Role: {message.role}")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"\nContent:")
    for content_block in message.content:
        if hasattr(content_block, "text"):
            print(f"  {content_block.text}")
    print()


def system_prompt_example():
    """Example with system prompt."""
    print("=" * 60)
    print("System Prompt Example")
    print("=" * 60)

    # Create client with system message echo enabled
    config = ParrotConfig(echo_system_messages=True)
    client = Anthropic(parrot_config=config)

    # Create message with system prompt
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant specialized in Python programming.",
        messages=[{"role": "user", "content": "How do I create a list in Python?"}],
    )

    print("Response with system prompt:")
    for content_block in message.content:
        if hasattr(content_block, "text"):
            print(f"  {content_block.text}")
    print()


def streaming_example():
    """Streaming response example."""
    print("=" * 60)
    print("Streaming Example")
    print("=" * 60)

    client = Anthropic()

    # Stream the response
    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Tell me a story about a parrot."}],
        stream=True,
    )

    print("Streaming response:\n")
    for event in stream:
        # Handle different event types
        if event.type == "message_start":
            print(f"[Message started: {event.message.id}]")
        elif event.type == "content_block_start":
            print("[Content block started]")
        elif event.type == "content_block_delta":
            # Print text deltas
            if hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop":
            print("\n[Content block stopped]")
        elif event.type == "message_delta":
            print(f"[Message completed: {event.delta.stop_reason}]")
        elif event.type == "message_stop":
            print("[Stream finished]")

    print()


async def async_example():
    """Async message creation example."""
    print("=" * 60)
    print("Async Example")
    print("=" * 60)

    # Create async client
    client = AsyncAnthropic()

    # Make async request
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    print("Async response:")
    for content_block in message.content:
        if hasattr(content_block, "text"):
            print(f"  {content_block.text}")
    print()


async def async_streaming_example():
    """Async streaming example."""
    print("=" * 60)
    print("Async Streaming Example")
    print("=" * 60)

    client = AsyncAnthropic()

    # Stream async response
    stream = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
    )

    print("Async streaming:\n")
    async for event in stream:
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)

    print("\n")


def tool_use_example():
    """Example with tool use using Parrot's special syntax."""
    print("=" * 60)
    print("Tool Use Example")
    print("=" * 60)

    client = Anthropic()

    # Use Parrot's tool call encoding syntax: [TOOL:function_name|param=value]
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Get weather data [TOOL:get_weather|city=London|units=metric]",
            }
        ],
    )

    print(f"Stop Reason: {message.stop_reason}")
    print(f"\nContent Blocks ({len(message.content)}):")

    for i, content_block in enumerate(message.content):
        print(f"\n  Block {i + 1}:")
        if hasattr(content_block, "text"):
            print(f"    Type: text")
            print(f"    Text: {content_block.text}")
        elif hasattr(content_block, "name"):
            print(f"    Type: tool_use")
            print(f"    ID: {content_block.id}")
            print(f"    Name: {content_block.name}")
            print(f"    Input: {content_block.input}")

    print()


def multi_turn_example():
    """Example with multiple messages."""
    print("=" * 60)
    print("Multi-turn Conversation Example")
    print("=" * 60)

    client = Anthropic()

    # Multi-turn conversation
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hi! My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"},
        ],
    )

    print("Multi-turn response:")
    for content_block in message.content:
        if hasattr(content_block, "text"):
            print(f"  {content_block.text}")
    print()


def custom_config_example():
    """Example with custom ParrotConfig."""
    print("=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    # Create client with custom configuration
    config = ParrotConfig(
        max_tokens=15,  # Limit response length
        truncate_at_word=True,  # Truncate at word boundaries
    )

    client = Anthropic(parrot_config=config)

    # Long message that will be truncated
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "This is a very long message that will be truncated based on the token limit we configured in the ParrotConfig. It should stop after about 15 tokens to demonstrate the truncation feature.",
            }
        ],
    )

    print("Truncated response:")
    for content_block in message.content:
        if hasattr(content_block, "text"):
            print(f"  {content_block.text}")
    print("(Response was limited to ~15 tokens)\n")


def main():
    """Run all examples."""
    print("\n🦜 Parrot Model - Anthropic SDK Adapter Examples\n")

    # Synchronous examples
    basic_example()
    system_prompt_example()
    streaming_example()
    tool_use_example()
    multi_turn_example()
    custom_config_example()

    # Async examples
    print("Running async examples...\n")
    asyncio.run(async_example())
    asyncio.run(async_streaming_example())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
