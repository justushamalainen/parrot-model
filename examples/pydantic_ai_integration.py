"""
Example usage of ParrotModel with Pydantic AI.

This script demonstrates how to use the ParrotPydanticModel adapter
to integrate ParrotModel with Pydantic AI agents for local testing
and development.
"""

import asyncio

from pydantic import BaseModel

from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
from parrot_model.core.config import ParrotConfig


async def basic_example():
    """Basic example of using ParrotModel with Pydantic AI."""
    print("=" * 60)
    print("Example 1: Basic ParrotModel with Pydantic AI Agent")
    print("=" * 60)

    # Import Agent here to avoid issues if running standalone
    from pydantic_ai import Agent

    # Create a Parrot model
    model = ParrotPydanticModel(
        model_name="parrot-echo", config=ParrotConfig(max_tokens=100)
    )

    # Create an agent with the Parrot model
    agent = Agent(model)

    # Run the agent
    result = await agent.run("Hello, Pydantic AI!")

    print(f"Input: 'Hello, Pydantic AI!'")
    print(f"Response: {result.data}")
    print()


async def structured_output_example():
    """Example with structured output."""
    print("=" * 60)
    print("Example 2: Structured Output")
    print("=" * 60)

    from pydantic_ai import Agent

    # Define a structured output model
    class Greeting(BaseModel):
        message: str
        language: str

    # Create model and agent with structured output
    model = ParrotPydanticModel()
    agent = Agent(model, result_type=Greeting)

    # Note: Since ParrotModel echoes back, we need to format the input
    # to match the expected structure for it to work properly
    result = await agent.run('{"message": "Hello", "language": "English"}')

    print(f"Structured output type: {type(result.data)}")
    print(f"Message: {result.data.message if hasattr(result.data, 'message') else result.data}")
    print()


async def tool_example():
    """Example with tool calls."""
    print("=" * 60)
    print("Example 3: Tool Calls")
    print("=" * 60)

    from pydantic_ai import Agent, RunContext

    # Create model with tool calls enabled
    model = ParrotPydanticModel(config=ParrotConfig(enable_tool_calls=True))

    # Create agent
    agent = Agent(model)

    # Define a tool
    @agent.tool
    def get_weather(ctx: RunContext, city: str) -> str:
        """Get the weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"

    # To trigger a tool call, we use the ParrotModel tool syntax
    # [TOOL:function_name|param=value]
    result = await agent.run("Please check [TOOL:get_weather|city=London]")

    print(f"Input: 'Please check [TOOL:get_weather|city=London]'")
    print(f"Result: {result.data}")
    print()


async def streaming_example():
    """Example with streaming."""
    print("=" * 60)
    print("Example 4: Streaming Response")
    print("=" * 60)

    from pydantic_ai import Agent

    # Create model with custom stream delay
    model = ParrotPydanticModel(
        config=ParrotConfig(stream_delay_ms=20)  # Faster streaming for demo
    )

    # Create agent
    agent = Agent(model)

    # Stream the response
    print("Streaming response: ", end="", flush=True)
    async with agent.run_stream("Hello, streaming world!") as result:
        async for message in result.stream_text():
            print(message, end="", flush=True)
    print("\n")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Parrot Model + Pydantic AI Examples" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    await basic_example()
    await structured_output_example()
    await tool_example()
    await streaming_example()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
