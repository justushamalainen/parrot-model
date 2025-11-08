"""
Streaming examples for the Parrot Model.

This script demonstrates the streaming capabilities of ParrotModel,
including synchronous and asynchronous streaming with various chunk
sizes and delay configurations.
"""

import asyncio
import time

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig


def example_basic_streaming():
    """Basic example of synchronous streaming."""
    print("=" * 60)
    print("Example 1: Basic Synchronous Streaming")
    print("=" * 60)

    # Create a model with default streaming settings
    model = ParrotModel()

    message = "Hello, streaming world!"
    print(f"Message: {message}\n")
    print("Streamed output: ", end="", flush=True)

    # Stream the response character by character
    for chunk in model.stream(message):
        print(chunk, end="", flush=True)

    print("\n")


def example_custom_chunk_size():
    """Example demonstrating different chunk sizes."""
    print("=" * 60)
    print("Example 2: Custom Chunk Sizes")
    print("=" * 60)

    model = ParrotModel()
    message = "Streaming with different chunk sizes demonstrates flexibility"

    # Stream with chunk_size=1 (character by character)
    print("Chunk size = 1 (character by character):")
    print("  ", end="", flush=True)
    for chunk in model.stream(message, chunk_size=1):
        print(chunk, end="", flush=True)
    print()

    # Stream with chunk_size=5
    print("\nChunk size = 5:")
    chunks = list(model.stream(message, chunk_size=5))
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: '{chunk}'")

    # Stream with chunk_size=10
    print("\nChunk size = 10:")
    chunks = list(model.stream(message, chunk_size=10))
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: '{chunk}'")

    print()


def example_custom_delay():
    """Example showing custom stream delay configuration."""
    print("=" * 60)
    print("Example 3: Custom Stream Delays")
    print("=" * 60)

    message = "Custom delays control streaming speed"

    # Fast streaming (10ms delay)
    print("Fast streaming (10ms delay):")
    config_fast = ParrotConfig(stream_delay_ms=10)
    model_fast = ParrotModel(config=config_fast)

    print("  ", end="", flush=True)
    start = time.time()
    for chunk in model_fast.stream(message, chunk_size=5):
        print(chunk, end="", flush=True)
    fast_time = time.time() - start
    print(f" (took {fast_time:.2f}s)")

    # Slow streaming (100ms delay)
    print("\nSlow streaming (100ms delay):")
    config_slow = ParrotConfig(stream_delay_ms=100)
    model_slow = ParrotModel(config=config_slow)

    print("  ", end="", flush=True)
    start = time.time()
    for chunk in model_slow.stream(message, chunk_size=5):
        print(chunk, end="", flush=True)
    slow_time = time.time() - start
    print(f" (took {slow_time:.2f}s)")

    print()


async def example_async_streaming():
    """Example of asynchronous streaming."""
    print("=" * 60)
    print("Example 4: Async Streaming")
    print("=" * 60)

    # Create a model
    model = ParrotModel()

    message = "Async streaming allows non-blocking operations"
    print(f"Message: {message}\n")
    print("Async streamed output: ", end="", flush=True)

    # Stream asynchronously
    async for chunk in model.astream(message):
        print(chunk, end="", flush=True)

    print("\n")


async def example_concurrent_streams():
    """Example running multiple streams concurrently."""
    print("=" * 60)
    print("Example 5: Concurrent Async Streams")
    print("=" * 60)

    # Create a model with faster streaming for demo
    config = ParrotConfig(stream_delay_ms=10)
    model = ParrotModel(config=config)

    # Define messages for concurrent streaming
    messages = [
        "Stream one",
        "Stream two",
        "Stream three"
    ]

    # Helper function to stream and collect
    async def stream_and_collect(msg: str, stream_id: int) -> str:
        result = []
        async for chunk in model.astream(msg, chunk_size=3):
            result.append(chunk)
        return f"Stream {stream_id}: {''.join(result)}"

    # Run all streams concurrently
    print("Streaming 3 messages concurrently...\n")
    results = await asyncio.gather(
        *[stream_and_collect(msg, i+1) for i, msg in enumerate(messages)]
    )

    # Display results
    for result in results:
        print(f"  {result}")

    print()


def example_streaming_with_truncation():
    """Example showing streaming with text truncation."""
    print("=" * 60)
    print("Example 6: Streaming with Text Truncation")
    print("=" * 60)

    # Create a model with token limit
    config = ParrotConfig(max_tokens=8, truncate_at_word=True)
    model = ParrotModel(config=config)

    message = "This is a very long message that will be truncated during streaming"
    print(f"Original message: {message}")
    print(f"Config: max_tokens={config.max_tokens}, truncate_at_word={config.truncate_at_word}\n")

    print("Streamed output (truncated): ", end="", flush=True)
    for chunk in model.stream(message, chunk_size=5):
        print(chunk, end="", flush=True)

    print("\n")


async def example_async_streaming_word_by_word():
    """Example of word-by-word async streaming."""
    print("=" * 60)
    print("Example 7: Word-by-Word Async Streaming")
    print("=" * 60)

    # Create a model with custom delay
    config = ParrotConfig(stream_delay_ms=200)  # Slower to see word-by-word
    model = ParrotModel(config=config)

    message = "This demonstrates word by word streaming output"
    print(f"Message: {message}\n")
    print("Streaming words: ", end="", flush=True)

    # Simulate word-by-word streaming by using the word count as chunk size
    words = message.split()
    current_pos = 0

    async for chunk in model.astream(message):
        print(chunk, end="", flush=True)
        current_pos += len(chunk)

        # Add space after each word
        if current_pos < len(message) and message[current_pos:current_pos+1] == " ":
            await asyncio.sleep(0.2)  # Pause between words

    print("\n")


def example_streaming_progress():
    """Example showing progress tracking during streaming."""
    print("=" * 60)
    print("Example 8: Streaming with Progress Tracking")
    print("=" * 60)

    model = ParrotModel()
    message = "Track progress while streaming responses"

    print(f"Message: {message}")
    print(f"Message length: {len(message)} characters\n")

    total_streamed = 0
    chunk_count = 0

    print("Progress:")
    for chunk in model.stream(message, chunk_size=5):
        chunk_count += 1
        total_streamed += len(chunk)
        progress = (total_streamed / len(message)) * 100
        print(f"  Chunk {chunk_count}: '{chunk}' (Progress: {progress:.1f}%)")

    print(f"\nTotal chunks: {chunk_count}")
    print(f"Total characters streamed: {total_streamed}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 16 + "Parrot Model - Streaming" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Synchronous examples
    example_basic_streaming()
    example_custom_chunk_size()
    example_custom_delay()
    example_streaming_with_truncation()
    example_streaming_progress()

    # Async examples
    asyncio.run(example_async_streaming())
    asyncio.run(example_concurrent_streams())
    asyncio.run(example_async_streaming_word_by_word())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
