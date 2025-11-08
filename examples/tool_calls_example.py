"""
Tool call examples for the Parrot Model.

This script demonstrates how to use the tool call parsing and handling
features of the Parrot Model, including encoding, parsing, and retrieving
tool calls from messages.
"""

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig
from parrot_model.core.tools import ToolCallParser


def example_basic_tool_call():
    """Basic example of encoding and parsing tool calls."""
    print("=" * 60)
    print("Example 1: Basic Tool Call")
    print("=" * 60)

    # Create a model with tool calls enabled (default)
    model = ParrotModel()

    # Send a message with a tool call using the syntax:
    # [TOOL:function_name|param1=value1|param2=value2]
    message = "What's the weather like in [TOOL:get_weather|city=London|units=metric]?"
    print(f"User: {message}")

    # Generate response - tool calls are parsed automatically
    response = model.generate(message)
    print(f"Model: {response}")

    # Retrieve the parsed tool calls
    tool_calls = model.get_tool_calls()
    print(f"\nTool calls detected: {len(tool_calls)}")

    for tc in tool_calls:
        print(f"  Function: {tc.name}")
        print(f"  Parameters: {tc.parameters}")
        print(f"  Call ID: {tc.id}")
    print()


def example_multiple_tool_calls():
    """Example with multiple tool calls in a single message."""
    print("=" * 60)
    print("Example 2: Multiple Tool Calls")
    print("=" * 60)

    model = ParrotModel()

    # Message with multiple tool calls
    message = (
        "Please [TOOL:get_weather|city=London] and [TOOL:get_weather|city=Paris] "
        "and then [TOOL:compare_data|city1=London|city2=Paris]"
    )
    print(f"User: {message}\n")

    response = model.generate(message)
    print(f"Model: {response}")

    # Get all tool calls
    tool_calls = model.get_tool_calls()
    print(f"\nTool calls detected: {len(tool_calls)}")

    for i, tc in enumerate(tool_calls, 1):
        print(f"\nTool Call {i}:")
        print(f"  Function: {tc.name}")
        print(f"  Parameters: {tc.parameters}")
        print(f"  ID: {tc.id}")
    print()


def example_encoding_tool_calls():
    """Example of programmatically encoding tool calls."""
    print("=" * 60)
    print("Example 3: Programmatic Tool Call Encoding")
    print("=" * 60)

    # Create tool call encodings programmatically using ToolCallParser.encode()
    weather_call = ToolCallParser.encode("get_weather", city="London", units="metric")
    print(f"Encoded weather call: {weather_call}")

    search_call = ToolCallParser.encode(
        "search_database",
        table="users",
        query="active=true",
        limit="100"
    )
    print(f"Encoded search call: {search_call}")

    # Use them in a message
    model = ParrotModel()
    message = f"First check {weather_call}, then run {search_call}"
    print(f"\nCombined message: {message}\n")

    response = model.generate(message)
    print(f"Response: {response}")

    # Verify parsed tool calls
    tool_calls = model.get_tool_calls()
    print(f"\nDetected {len(tool_calls)} tool calls:")
    for tc in tool_calls:
        print(f"  - {tc.name}: {tc.parameters}")
    print()


def example_parsing_tool_calls():
    """Example of manually using the ToolCallParser."""
    print("=" * 60)
    print("Example 4: Manual Tool Call Parsing")
    print("=" * 60)

    # Create a parser with the default pattern
    parser = ToolCallParser(r"\[TOOL:(\w+)\|(.+?)\]")

    # Parse tool calls from a message
    message = "Execute [TOOL:send_email|to=user@example.com|subject=Hello] now"
    tool_calls = parser.parse(message)

    print(f"Message: {message}")
    print(f"\nParsed {len(tool_calls)} tool call(s):")
    for tc in tool_calls:
        print(f"  Name: {tc.name}")
        print(f"  Parameters: {tc.parameters}")

    # Remove tool calls from message
    clean_message = parser.remove_from_message(message)
    print(f"\nMessage without tool calls: {clean_message}")
    print()


def example_tool_calls_disabled():
    """Example showing behavior with tool calls disabled."""
    print("=" * 60)
    print("Example 5: Tool Calls Disabled")
    print("=" * 60)

    # Create a model with tool calls disabled
    config = ParrotConfig(enable_tool_calls=False)
    model = ParrotModel(config=config)

    message = "Check [TOOL:get_weather|city=London] please"
    print(f"Config: enable_tool_calls=False")
    print(f"User: {message}")

    # With tool calls disabled, the syntax is treated as plain text
    response = model.generate(message)
    print(f"Model: {response}")
    print(f"Tool calls: {model.get_tool_calls()}")  # Will be empty

    print("\n" + "-" * 60 + "\n")

    # Compare with tool calls enabled
    model_enabled = ParrotModel()  # Default has tool calls enabled
    print(f"Config: enable_tool_calls=True")
    print(f"User: {message}")

    response = model_enabled.generate(message)
    print(f"Model: {response}")
    print(f"Tool calls: {len(model_enabled.get_tool_calls())} detected")
    print()


def example_tool_call_workflow():
    """Example of a typical workflow with tool calls."""
    print("=" * 60)
    print("Example 6: Typical Tool Call Workflow")
    print("=" * 60)

    model = ParrotModel()

    # Step 1: User sends a message with a tool call
    message = "What's the weather in [TOOL:get_weather|city=San Francisco|units=imperial]?"
    print(f"Step 1 - User message:\n  {message}\n")

    # Step 2: Generate response and get tool calls
    response = model.generate(message)
    print(f"Step 2 - Model response:\n  {response}\n")

    # Step 3: Process tool calls
    tool_calls = model.get_tool_calls()
    print(f"Step 3 - Processing {len(tool_calls)} tool call(s)...\n")

    for tc in tool_calls:
        print(f"Tool Call Details:")
        print(f"  ID: {tc.id}")
        print(f"  Function: {tc.name}")
        print(f"  Parameters: {tc.parameters}")

        # In a real application, you would:
        # 1. Execute the tool with these parameters
        # 2. Get the result from the tool
        # 3. Send the result back in a follow-up message

        # Simulated tool execution
        if tc.name == "get_weather":
            city = tc.parameters.get("city", "Unknown")
            units = tc.parameters.get("units", "metric")
            temp = "72°F" if units == "imperial" else "22°C"
            print(f"\n  Simulated Tool Result:")
            print(f"    Weather in {city}: Sunny, {temp}")

    print()


def example_complex_parameters():
    """Example with complex parameter values."""
    print("=" * 60)
    print("Example 7: Complex Parameter Values")
    print("=" * 60)

    # Encode tool calls with various parameter types
    tool_call = ToolCallParser.encode(
        "search_products",
        query="laptop computer",
        price_range="500-1500",
        categories="electronics,computers",
        sort_by="price_desc"
    )

    print(f"Encoded tool call:\n  {tool_call}\n")

    # Use it in a message
    model = ParrotModel()
    message = f"Find me some options: {tool_call}"
    response = model.generate(message)

    # Retrieve and display parsed parameters
    tool_calls = model.get_tool_calls()
    if tool_calls:
        tc = tool_calls[0]
        print(f"Parsed tool call:")
        print(f"  Function: {tc.name}")
        print(f"  Parameters:")
        for key, value in sorted(tc.parameters.items()):
            print(f"    {key}: {value}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 14 + "Parrot Model - Tool Calls" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    example_basic_tool_call()
    example_multiple_tool_calls()
    example_encoding_tool_calls()
    example_parsing_tool_calls()
    example_tool_calls_disabled()
    example_tool_call_workflow()
    example_complex_parameters()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
