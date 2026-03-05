"""これは自動生成されたファイルです。修正しないでください。

このファイルは `langgraph-gen` バージョン 0.0.3 を使用して生成されました。
このファイルを再生成するには、`langgraph-gen` をソース `yaml` ファイルを引き数として実行してください。

使用方法:

1. 生成されたファイルをプロジェクトに追加します。
2. スタブを使用して新しいエージェントを作成します。

以下は生成されたスタブのサンプル実装です:

```python
from typing_extensions import TypedDict

from stub import CustomAgent

class TrafficState(TypedDict):
    # ここに属性を定義してください
    foo: str

# スタンドアロン関数を定義
def chatbot(state: TrafficState) -> dict:
    print("In node: chatbot")
    return {
        # ここに状態更新ロジックを追加してください
    }


def visualization(state: TrafficState) -> dict:
    print("In node: visualization")
    return {
        # ここに状態更新ロジックを追加してください
    }


def analyze_with_llm(state: TrafficState) -> dict:
    print("In node: analyze_with_llm")
    return {
        # ここに状態更新ロジックを追加してください
    }


def report(state: TrafficState) -> dict:
    print("In node: report")
    return {
        # ここに状態更新ロジックを追加してください
    }


agent = CustomAgent(
    state_schema=TrafficState,
    impl=[
        ("chatbot", chatbot),
        ("visualization", visualization),
        ("analyze_with_llm", analyze_with_llm),
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
    """CustomAgentのステートグラフを作成します。"""
    # ステートグラフを宣言
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "chatbot",
        "visualization",
        "analyze_with_llm",  # 替代cluster和draw_hulls
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

    # ノードを追加
    builder.add_node("chatbot", nodes_by_name["chatbot"])
    builder.add_node("visualization", nodes_by_name["visualization"])
    builder.add_node("analyze_with_llm", nodes_by_name["analyze_with_llm"])
    builder.add_node("report", nodes_by_name["report"])

    # エッジを追加
    builder.add_edge(START, "chatbot")
    builder.add_edge("report", END)
    builder.add_edge("chatbot", "visualization")
    builder.add_edge("visualization", "analyze_with_llm")  # 替代cluster和draw_hulls
    builder.add_edge("analyze_with_llm", "report")
    return builder
