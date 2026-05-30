"""This file was generated using `langgraph-gen` version 0.0.3.

This file provides a placeholder implementation for the corresponding stub.

Replace the placeholder implementation with your own logic.
"""

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
    ],
)

compiled_agent = agent.compile()

print(compiled_agent.invoke({"foo": "bar"}))
