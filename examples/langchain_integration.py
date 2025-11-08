"""
LangChain and LangGraph integration examples for Parrot Model.

This script demonstrates how to use ParrotChatModel with LangChain's
LCEL (LangChain Expression Language), chains, and LangGraph workflows.
"""

import asyncio

# Optional imports - handle gracefully if not installed
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain is not installed. Install with: pip install langchain-core")
    print("Some examples will be skipped.\n")

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from parrot_model.adapters.langchain import ParrotChatModel
from parrot_model.core.config import ParrotConfig


def example_basic_chat_model():
    """Basic example using ParrotChatModel."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 1: Basic ChatModel Usage")
    print("=" * 60)

    # Create the model
    model = ParrotChatModel()

    # Create messages
    messages = [
        HumanMessage(content="Hello, LangChain!")
    ]

    # Invoke the model
    response = model.invoke(messages)

    print(f"Input: {messages[0].content}")
    print(f"Output: {response.content}")
    print(f"Type: {type(response)}")
    print()


def example_system_messages():
    """Example with system and multiple messages."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 2: System and Multiple Messages")
    print("=" * 60)

    # By default, system messages are not echoed
    model = ParrotChatModel()

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="The answer is 4."),
        HumanMessage(content="What about 3+3?")
    ]

    response = model.invoke(messages)

    print("Messages:")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content}")

    print(f"\nOutput: {response.content}")
    print("(Note: System message is excluded by default)")
    print()


def example_lcel_chain():
    """Example using LCEL (LangChain Expression Language)."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 3: LCEL Chain")
    print("=" * 60)

    # Create a model
    model = ParrotChatModel()

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])

    # Create an output parser
    parser = StrOutputParser()

    # Build a chain using LCEL
    chain = prompt | model | parser

    # Invoke the chain
    result = chain.invoke({"input": "Hello from LCEL chain!"})

    print(f"Input: Hello from LCEL chain!")
    print(f"Output: {result}")
    print(f"Type: {type(result)}")
    print()


def example_streaming():
    """Example of streaming with LangChain."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 4: Streaming")
    print("=" * 60)

    # Create a model with faster streaming for demo
    config = ParrotConfig(stream_delay_ms=20)
    model = ParrotChatModel(parrot_config=config)

    messages = [HumanMessage(content="Stream this message please!")]

    print(f"Input: {messages[0].content}")
    print("Streamed output: ", end="", flush=True)

    # Stream the response
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)

    print("\n")


async def example_async_streaming():
    """Example of async streaming with LangChain."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 5: Async Streaming")
    print("=" * 60)

    # Create a model
    config = ParrotConfig(stream_delay_ms=20)
    model = ParrotChatModel(parrot_config=config)

    messages = [HumanMessage(content="Async streaming in LangChain")]

    print(f"Input: {messages[0].content}")
    print("Async streamed output: ", end="", flush=True)

    # Stream asynchronously
    async for chunk in model.astream(messages):
        print(chunk.content, end="", flush=True)

    print("\n")


def example_tool_calls():
    """Example with tool calls in LangChain."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 6: Tool Calls")
    print("=" * 60)

    # Create a model with tool calls enabled
    config = ParrotConfig(enable_tool_calls=True)
    model = ParrotChatModel(parrot_config=config)

    # Send a message with a tool call using Parrot's syntax
    messages = [
        HumanMessage(content="Please [TOOL:get_weather|city=London|units=metric]")
    ]

    response = model.invoke(messages)

    print(f"Input: {messages[0].content}")
    print(f"Output: {response.content}")

    # Check for tool calls in additional_kwargs
    if "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        print(f"\nTool calls detected: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"  Function: {tc['function']['name']}")
            print(f"  Arguments: {tc['function']['arguments']}")
            print(f"  ID: {tc['id']}")
    print()


def example_configuration():
    """Example showing various configuration options."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 7: Custom Configuration")
    print("=" * 60)

    # Create a custom configuration
    config = ParrotConfig(
        max_tokens=15,
        truncate_at_word=True,
        stream_delay_ms=10,
        enable_tool_calls=True
    )

    model = ParrotChatModel(parrot_config=config)

    messages = [
        HumanMessage(content="This is a very long message that will be truncated by the configuration")
    ]

    response = model.invoke(messages)

    print(f"Config: max_tokens={config.max_tokens}, truncate_at_word={config.truncate_at_word}")
    print(f"Input: {messages[0].content}")
    print(f"Output: {response.content}")
    print()


def example_langgraph_simple():
    """Simple example using LangGraph."""
    if not LANGCHAIN_AVAILABLE or not LANGGRAPH_AVAILABLE:
        if not LANGGRAPH_AVAILABLE:
            print("=" * 60)
            print("Example 8: LangGraph (Skipped)")
            print("=" * 60)
            print("LangGraph is not installed. Install with: pip install langgraph")
            print()
        return

    print("=" * 60)
    print("Example 8: Simple LangGraph Workflow")
    print("=" * 60)

    # Create a model
    model = ParrotChatModel()

    # Define state
    class State(dict):
        pass

    # Define nodes
    def process_input(state: State) -> State:
        """Process the input message."""
        user_input = state["input"]
        print(f"  Processing: {user_input}")
        state["processed"] = f"Processed: {user_input}"
        return state

    def generate_response(state: State) -> State:
        """Generate response using the model."""
        messages = [HumanMessage(content=state["processed"])]
        response = model.invoke(messages)
        print(f"  Generated: {response.content}")
        state["output"] = response.content
        return state

    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("process", process_input)
    workflow.add_node("generate", generate_response)

    # Define the flow
    workflow.set_entry_point("process")
    workflow.add_edge("process", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    input_text = "Hello from LangGraph!"
    print(f"Input: {input_text}\n")
    result = app.invoke({"input": input_text})

    print(f"\nFinal output: {result['output']}")
    print()


async def example_async_batch():
    """Example of async batch processing."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 9: Async Batch Processing")
    print("=" * 60)

    # Create a model
    model = ParrotChatModel()

    # Create multiple message sets
    batch = [
        [HumanMessage(content="First message")],
        [HumanMessage(content="Second message")],
        [HumanMessage(content="Third message")]
    ]

    print("Processing 3 messages in batch...\n")

    # Process batch asynchronously
    results = await model.abatch(batch)

    for i, result in enumerate(results, 1):
        print(f"{i}. Input: {batch[i-1][0].content}")
        print(f"   Output: {result.content}")

    print()


def example_chain_with_parser():
    """Example of a chain with output parsing."""
    if not LANGCHAIN_AVAILABLE:
        return

    print("=" * 60)
    print("Example 10: Chain with Output Parser")
    print("=" * 60)

    # Create components
    model = ParrotChatModel()
    prompt = ChatPromptTemplate.from_template("Process this: {text}")
    parser = StrOutputParser()

    # Build chain
    chain = prompt | model | parser

    # Invoke chain
    result = chain.invoke({"text": "Important data"})

    print(f"Input: Important data")
    print(f"Output: {result}")
    print(f"Type: {type(result)}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 11 + "Parrot Model - LangChain Integration" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    if not LANGCHAIN_AVAILABLE:
        print("Please install LangChain to run these examples:")
        print("  pip install parrot-model[langchain]")
        print()
        return

    # Synchronous examples
    example_basic_chat_model()
    example_system_messages()
    example_lcel_chain()
    example_streaming()
    example_tool_calls()
    example_configuration()
    example_chain_with_parser()
    example_langgraph_simple()

    # Async examples
    asyncio.run(example_async_streaming())
    asyncio.run(example_async_batch())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
