# 交通拥堵预测ワークフロー (Traffic Congestion Prediction Workflow)

## 概要

これは交通拥堵予測分析システムのコアコードモジュールです。完全なデータ分析と予測機能を提供します。

## コア機能

| モジュール | 機能 |
|------|------|
| **Chatbot** | ユーザー入力を解析し、路線と時間情報を抽出 |
| **Visualization** | 拥堵イベントを三角形に変換して可視化 |
| **Cluster** | DBSCANクラスタリングにより拥堵ホットスポットを識別 |
| **Draw Hulls** | 外包三角形を描画し、拥堵エリアを表示 |
| **LLM Analysis** | LLMを使用して拥堵予測分析を実施 |
| **Report** | 分析レポートを生成 |

## クイックスタート

```python
from workflow.test_complete_workflow import test_complete_workflow

# 完全ワークフローを実行
result = test_complete_workflow(
    user_input="関越道2024年の交通拥堵情况を分析してください",
    ground_truth_file="data/ground_truth.csv"
)
```

## ファイル説明

| ファイル | 説明 |
|------|------|
| `functions.py` | コア機能関数（データ処理、クラスタリング、可視化） |
| `implementation.py` | LangGraphワークフロー定義 |
| `config.py` | 設定管理 |
| `stub.py` | ワークフローノード定義 |
| `llm_analyzer.py` | LLM分析と予測 |
| `prompt_templates.py` | プロンプトテンプレート |
| `data_preparers.py` | データ準備 |
| `shape_postprocessor.py` | 形状後処理 |
| `evaluation.py` | 評価機能 |
| `visualization_*.py` | 可視化機能 |
| `test_complete_workflow.py` | 完全ワークフローテスト |

## 依存関係

- pandas, numpy, matplotlib
- scikit-learn (DBSCANクラスタリング)
- openai / anthropic (LLM呼び出し)

## データディレクトリ

データファイルは `../data/` ディレクトリに保存されています
