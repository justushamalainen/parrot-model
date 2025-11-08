"""
OpenAI SDK Adapter Example

This example demonstrates how to use the Parrot Model as a drop-in replacement
for the OpenAI Python SDK. The ParrotModel mimics the OpenAI client interface,
allowing you to develop and test OpenAI-based applications without API keys or costs.

Features demonstrated:
- Basic chat completions
- Streaming responses
- Async operations
- Tool calls
- Custom configuration
"""

import asyncio

from parrot_model import ParrotConfig
from parrot_model.adapters import AsyncOpenAI, OpenAI


def basic_example():
    """Basic synchronous chat completion example."""
    print("=" * 60)
    print("Basic Example: OpenAI SDK Interface")
    print("=" * 60)

    # Create client - just like the real OpenAI SDK
    client = OpenAI()

    # Make a chat completion request
    response = client.chat.completions.create(
        model="gpt-4",  # Model name is for compatibility, ignored by Parrot
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )

    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Message: {response.choices[0].message.content}")
    print(f"Finish Reason: {response.choices[0].finish_reason}")
    print()


def streaming_example():
    """Streaming response example."""
    print("=" * 60)
    print("Streaming Example")
    print("=" * 60)

    client = OpenAI()

    # Stream the response
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Tell me a story about a parrot."}],
        stream=True,
    )

    print("Streaming response: ", end="", flush=True)
    for chunk in stream:
        # Access delta content from streaming chunks
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


async def async_example():
    """Async chat completion example."""
    print("=" * 60)
    print("Async Example")
    print("=" * 60)

    # Create async client
    client = AsyncOpenAI()

    # Make async request
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )

    print(f"Async Response: {response.choices[0].message.content}")
    print()


async def async_streaming_example():
    """Async streaming example."""
    print("=" * 60)
    print("Async Streaming Example")
    print("=" * 60)

    client = AsyncOpenAI()

    # Stream async response
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
    )

    print("Async streaming: ", end="", flush=True)
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def tool_call_example():
    """Example with tool calls using Parrot's special syntax."""
    print("=" * 60)
    print("Tool Call Example")
    print("=" * 60)

    client = OpenAI()

    # Use Parrot's tool call encoding syntax: [TOOL:function_name|param=value]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "Get the weather [TOOL:get_weather|city=London|units=metric]",
            }
        ],
    )

    message = response.choices[0].message

    print(f"Content: {message.content}")
    print(f"Finish Reason: {response.choices[0].finish_reason}")

    if message.tool_calls:
        print(f"\nTool Calls ({len(message.tool_calls)}):")
        for tool_call in message.tool_calls:
            print(f"  - ID: {tool_call['id']}")
            print(f"    Function: {tool_call['function']['name']}")
            print(f"    Arguments: {tool_call['function']['arguments']}")

    print()


def custom_config_example():
    """Example with custom ParrotConfig."""
    print("=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    # Create client with custom configuration
    config = ParrotConfig(
        max_tokens=20,  # Limit response length
        truncate_at_word=True,  # Truncate at word boundaries
        stream_delay_ms=100,  # Slower streaming for demo
    )

    client = OpenAI(parrot_config=config)

    # Long message that will be truncated
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "This is a very long message that will be truncated based on the token limit we configured. It should stop after about 20 tokens.",
            }
        ],
    )

    print(f"Truncated response: {response.choices[0].message.content}")
    print("(Response was limited to ~20 tokens)")
    print()


def main():
    """Run all examples."""
    print("\n🦜 Parrot Model - OpenAI SDK Adapter Examples\n")

    # Synchronous examples
    basic_example()
    streaming_example()
    tool_call_example()
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
