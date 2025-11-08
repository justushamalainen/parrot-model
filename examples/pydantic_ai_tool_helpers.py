"""
Example demonstrating the use of Pydantic AI tool call helpers.

This example shows how to use the helper functions to easily create
tool calls for Pydantic AI agents.
"""

from parrot_model.adapters.pydantic_ai_helpers import (
    create_tool_call,
    encode_tool_call_in_prompt,
)

try:
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from parrot_model.adapters.pydantic_ai import ParrotPydanticModel
except ImportError:
    print("Please install pydantic-ai to run this example:")
    print("  pip install pydantic-ai")
    exit(1)


def example_single_tool_call():
    """Example: Creating a single tool call."""
    print("\n=== Example 1: Single Tool Call ===\n")

    # Create a tool call using the helper
    tool_call = create_tool_call("get_weather", city="London", units="metric")

    print(f"Tool name: {tool_call.tool_name}")
    print(f"Arguments: {tool_call.args}")
    print(f"Tool call ID: {tool_call.tool_call_id}")

    # Use it in a ModelRequest
    request = ModelRequest(
        parts=[UserPromptPart(content="Check the weather"), tool_call]
    )
    print(f"\nCreated request with {len(request.parts)} parts")


def example_multiple_tool_calls():
    """Example: Creating multiple tool calls with list comprehension."""
    print("\n=== Example 2: Multiple Tool Calls ===\n")

    # Create multiple tool calls using list comprehension
    calls = [
        ("get_weather", {"city": "London"}),
        ("get_weather", {"city": "Paris"}),
        ("get_time", {"timezone": "UTC"}),
    ]
    tool_calls = [create_tool_call(name, **params) for name, params in calls]

    print(f"Created {len(tool_calls)} tool calls:")
    for i, tc in enumerate(tool_calls, 1):
        print(f"  {i}. {tc.tool_name} with args: {tc.args}")

    # Use them in a ModelRequest
    request = ModelRequest(
        parts=[UserPromptPart(content="Check multiple locations"), *tool_calls]
    )
    print(f"\nCreated request with {len(request.parts)} parts")


def example_encoded_in_prompt():
    """Example: Encoding tool calls in prompts for Parrot Model."""
    print("\n=== Example 3: Encoded Tool Calls in Prompts ===\n")

    # Create encoded tool call strings
    london_weather = encode_tool_call_in_prompt("get_weather", city="London")
    paris_weather = encode_tool_call_in_prompt("get_weather", city="Paris")

    print(f"London weather encoding: {london_weather}")
    print(f"Paris weather encoding: {paris_weather}")

    # Use in a prompt that will trigger tool calls in Parrot Model
    prompt = f"Compare {london_weather} and {paris_weather}"
    print(f"\nFull prompt: {prompt}")

    # Create request
    request = ModelRequest(parts=[UserPromptPart(content=prompt)])
    print(f"Created request that will trigger tool calls when processed by Parrot Model")


def example_with_agent():
    """Example: Using tool calls with Pydantic AI Agent."""
    print("\n=== Example 4: With Pydantic AI Agent ===\n")

    # Create Parrot model
    model = ParrotPydanticModel()

    # Create an agent
    agent = Agent(model)

    # Define a tool
    @agent.tool
    def get_weather(city: str, units: str = "metric") -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F ({units})"

    # Use encoded tool call in prompt
    tool_encoding = encode_tool_call_in_prompt(
        "get_weather", city="London", units="metric"
    )
    prompt = f"Please check {tool_encoding}"

    print(f"Prompt with tool encoding: {prompt}")
    print("\nRunning agent with tool call...")

    # Run the agent (the Parrot Model will parse the tool call)
    result = agent.run_sync(prompt)
    print(f"Result: {result.data}")


def example_custom_ids():
    """Example: Using custom tool call IDs."""
    print("\n=== Example 5: Custom Tool Call IDs ===\n")

    # Create tool calls with custom IDs
    tool_call_1 = create_tool_call(
        "database_query", tool_call_id="query-001", table="users", limit=10
    )
    tool_call_2 = create_tool_call(
        "database_query", tool_call_id="query-002", table="orders", limit=5
    )

    print(f"Tool call 1 ID: {tool_call_1.tool_call_id}")
    print(f"Tool call 2 ID: {tool_call_2.tool_call_id}")
    print("\nCustom IDs are useful for tracking specific tool calls in your application")


if __name__ == "__main__":
    print("=" * 60)
    print("Pydantic AI Tool Call Helpers - Examples")
    print("=" * 60)

    example_single_tool_call()
    example_multiple_tool_calls()
    example_encoded_in_prompt()
    example_with_agent()
    example_custom_ids()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
