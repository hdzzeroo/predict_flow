"""This is an automatically generated file. Do not modify it.

This file was generated using `langgraph-gen` version 0.0.3.
To regenerate this file, run `langgraph-gen` with the source `yaml` file as an argument.

Usage:

1. Add the generated file to your project.
2. Create a new agent using the stub.

Below is a sample implementation of the generated stub:

```python
from typing_extensions import TypedDict

from stub import CustomAgent

class TrafficState(TypedDict):
    # define your attributes here
    foo: str

# Define stand-alone functions
def chatbot(state: TrafficState) -> dict:
    print("In node: chatbot")
    return {
        # Add your state update logic here
    }


def visualization(state: TrafficState) -> dict:
    print("In node: visualization")
    return {
        # Add your state update logic here
    }


def cluster(state: TrafficState) -> dict:
    print("In node: cluster")
    return {
        # Add your state update logic here
    }


def draw_hulls(state: TrafficState) -> dict:
    print("In node: draw_hulls")
    return {
        # Add your state update logic here
    }


def report(state: TrafficState) -> dict:
    print("In node: report")
    return {
        # Add your state update logic here
    }


agent = CustomAgent(
    state_schema=TrafficState,
    impl=[
        ("chatbot", chatbot),
        ("visualization", visualization),
        ("cluster", cluster),
        ("draw_hulls", draw_hulls),
        ("report", report),
    ]
)

compiled_agent = agent.compile()

print(compiled_agent.invoke({"foo": "bar"}))
"""

from typing import Callable, Any, Optional, Type

from langgraph.constants import START, END
from langgraph.graph import StateGraph


def CustomAgent(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for CustomAgent."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "chatbot",
        "visualization",
        "cluster",
        "draw_hulls",
        "report",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("chatbot", nodes_by_name["chatbot"])
    builder.add_node("visualization", nodes_by_name["visualization"])
    builder.add_node("cluster", nodes_by_name["cluster"])
    builder.add_node("draw_hulls", nodes_by_name["draw_hulls"])
    builder.add_node("report", nodes_by_name["report"])

    # Add edges
    builder.add_edge(START, "chatbot")
    builder.add_edge("report", END)
    builder.add_edge("chatbot", "visualization")
    builder.add_edge("visualization", "cluster")
    builder.add_edge("cluster", "draw_hulls")
    builder.add_edge("draw_hulls", "report")
    return builder
