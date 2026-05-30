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

class SomeState(TypedDict):
    # define your attributes here
    foo: str

# Define stand-alone functions
def extract_file(state: SomeState) -> dict:
    print("In node: extract_file")
    return {
        # Add your state update logic here
    }


def read_file(state: SomeState) -> dict:
    print("In node: read_file")
    return {
        # Add your state update logic here
    }


def draw_trangle(state: SomeState) -> dict:
    print("In node: draw_trangle")
    return {
        # Add your state update logic here
    }


agent = CustomAgent(
    state_schema=SomeState,
    impl=[
        ("extract_file", extract_file),
        ("read_file", read_file),
        ("draw_trangle", draw_trangle),
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
        "extract_file",
        "read_file",
        "draw_trangle",
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
    builder.add_node("extract_file", nodes_by_name["extract_file"])
    builder.add_node("read_file", nodes_by_name["read_file"])
    builder.add_node("draw_trangle", nodes_by_name["draw_trangle"])

    # Add edges
    builder.add_edge(START, "extract_file")
    builder.add_edge("draw_trangle", END)
    builder.add_edge("extract_file", "read_file")
    builder.add_edge("read_file", "draw_trangle")
    return builder
