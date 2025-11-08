"""
Example demonstrating the use of LangChain/LangGraph tool call helpers.

This example shows how to use the helper functions to easily create
tool calls for LangChain chains and LangGraph workflows.
"""

from parrot_model.adapters.langchain_helpers import (
    create_tool_call,
    encode_tool_call_in_prompt,
)

try:
    from langchain_core.messages import AIMessage, HumanMessage

    from parrot_model.adapters.langchain import ParrotChatModel
except ImportError:
    print("Please install langchain-core to run this example:")
    print("  pip install langchain-core")
    exit(1)


def example_create_tool_call():
    """Example: Creating a ToolCall object using LangChain's native class."""
    print("\n=== Example 1: Single ToolCall Object ===\n")

    # Create a tool call using LangChain's native ToolCall class
    tool_call = create_tool_call("get_weather", city="London", units="metric")

    print(f"Tool call type: {type(tool_call).__name__}")
    print(f"Tool name: {tool_call.name}")
    print(f"Arguments: {tool_call.args}")
    print(f"Tool call ID: {tool_call.id}")

    # Use directly in an AIMessage
    message = AIMessage(content="Checking weather", tool_calls=[tool_call])
    print(f"\nCreated AIMessage with {len(message.tool_calls)} tool call(s)")


def example_create_multiple_tool_calls():
    """Example: Creating multiple ToolCall objects using list comprehension."""
    print("\n=== Example 2: Multiple ToolCall Objects ===\n")

    # Create multiple tool calls using list comprehension
    calls = [
        ("get_weather", {"city": "London"}),
        ("get_weather", {"city": "Paris"}),
        ("get_time", {"timezone": "UTC"}),
    ]
    tool_calls = [create_tool_call(name, **params) for name, params in calls]

    print(f"Created {len(tool_calls)} tool calls:")
    for i, tc in enumerate(tool_calls, 1):
        print(f"  {i}. {tc.name} with args: {tc.args}")

    # Use in an AIMessage
    message = AIMessage(content="Processing multiple requests", tool_calls=tool_calls)
    print(f"\nCreated AIMessage with {len(message.tool_calls)} tool call(s)")


def example_single_tool_call_message():
    """Example: Creating a message with a single tool call."""
    print("\n=== Example 3: Single Tool Call Message ===\n")

    # Create a tool call
    tool_call = create_tool_call("get_weather", city="London", units="metric")

    # Create an AI message with the tool call
    message = AIMessage(content="", tool_calls=[tool_call])

    print(f"Message type: {type(message).__name__}")
    print(f"Content: {message.content}")
    print(f"Tool calls: {len(message.tool_calls)}")

    # Create with content
    message_with_content = AIMessage(
        content="Let me check the weather for you",
        tool_calls=[create_tool_call("get_weather", city="London")]
    )

    print(f"\nMessage with content: {message_with_content.content}")


def example_multiple_tool_calls_message():
    """Example: Creating a message with multiple tool calls."""
    print("\n=== Example 4: Multiple Tool Calls Message ===\n")

    # Create multiple tool calls
    tool_calls = [
        create_tool_call("get_weather", city="London"),
        create_tool_call("get_weather", city="Paris"),
        create_tool_call("get_time", timezone="UTC"),
    ]

    # Create a message with multiple tool calls
    message = AIMessage(
        content="Checking multiple locations",
        tool_calls=tool_calls
    )

    print(f"Message content: {message.content}")
    print(f"Number of tool calls: {len(message.tool_calls)}")

    for i, tc in enumerate(message.tool_calls, 1):
        print(f"  {i}. {tc.name}: {tc.args}")


def example_encoded_in_prompt():
    """Example: Encoding tool calls in prompts for Parrot Model."""
    print("\n=== Example 5: Encoded Tool Calls in Prompts ===\n")

    # Create encoded tool call strings
    london_weather = encode_tool_call_in_prompt("get_weather", city="London")
    paris_weather = encode_tool_call_in_prompt("get_weather", city="Paris")

    print(f"London encoding: {london_weather}")
    print(f"Paris encoding: {paris_weather}")

    # Use in a message
    message = HumanMessage(content=f"Compare {london_weather} and {paris_weather}")
    print(f"\nFull message: {message.content}")


def example_with_chat_model():
    """Example: Using tool calls with ParrotChatModel."""
    print("\n=== Example 6: With ParrotChatModel ===\n")

    # Create the chat model
    model = ParrotChatModel()

    # Use encoded tool call in a message
    tool_encoding = encode_tool_call_in_prompt(
        "get_weather", city="London", units="metric"
    )
    message = HumanMessage(content=f"Check {tool_encoding} today")

    print(f"Input message: {message.content}")

    # Invoke the model
    response = model.invoke([message])

    print(f"\nResponse content: {response.content}")
    print(f"Tool calls in response: {response.additional_kwargs.get('tool_calls', [])}")


def example_in_message_list():
    """Example: Using tool calls in a conversation."""
    print("\n=== Example 7: In a Conversation ===\n")

    # Create a conversation with tool calls
    tool_call = create_tool_call("get_weather", city="London", units="metric")
    messages = [
        HumanMessage(content="What's the weather in London?"),
        AIMessage(
            content="Let me check that for you",
            tool_calls=[tool_call]
        ),
    ]

    print("Conversation messages:")
    for i, msg in enumerate(messages, 1):
        print(f"\n  Message {i}:")
        print(f"    Type: {type(msg).__name__}")
        print(f"    Content: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"    Tool calls: {len(msg.tool_calls)}")


def example_with_langchain_chain():
    """Example: Using in a LangChain chain."""
    print("\n=== Example 8: With LangChain Chain (LCEL) ===\n")

    try:
        from langchain_core.prompts import ChatPromptTemplate

        model = ParrotChatModel()

        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{input}")]
        )

        # Create a chain
        chain = prompt | model

        # Use with encoded tool call
        tool_encoding = encode_tool_call_in_prompt("calculate", x=5, y=3, op="add")
        response = chain.invoke({"input": f"Please {tool_encoding}"})

        print(f"Input: Please {tool_encoding}")
        print(f"Response: {response.content}")
        print(f"Tool calls: {response.additional_kwargs.get('tool_calls', [])}")

    except ImportError:
        print("Install langchain-core for full chain support")


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Tool Call Helpers - Examples")
    print("=" * 60)

    example_create_tool_call()
    example_create_multiple_tool_calls()
    example_single_tool_call_message()
    example_multiple_tool_calls_message()
    example_encoded_in_prompt()
    example_with_chat_model()
    example_in_message_list()
    example_with_langchain_chain()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
