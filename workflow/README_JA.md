# 交通拥堵预测工作流 - セットアップガイド

## 目次

1. [はじめに](#はじめに)
2. [環境要件](#環境要件)
3. [インストール](#インストール)
4. [データ準備](#データ準備)
5. [環境設定](#環境設定)
6. [クイックスタート](#クイックスタート)
7. [トラブルシューティング](#トラブルシューティング)

---

## はじめに

このプロジェクトは、日本の高速道路（関越道、東北道など）の交通拥堵を予測するシステムです。
LangGraphベースのワークフローとLLMを活用した分析を行います。

### 対応道路

- 関越道（KANNETSU）
- 東北道（TOUHOKU）
- アクアライン

---

## 環境要件

| 項目 | バージョン/要件 |
|------|-----------------|
| Python | 3.9 以上 |
| OS | macOS / Linux / Windows |

### 必要なライブラリ

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
shapely>=2.0.0
langgraph>=0.0.20
openai>=1.0.0
python-dotenv>=1.0.0
typing-extensions>=4.5.0
```

---

## インストール

### 1. リポジトリをクローン

```bash
git clone <リポジトリURL>
cd predict_workflow
```

### 2. 仮想環境の作成（推奨）

```bash
# venvの場合
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows

# condaの場合
conda create -n predict_workflow python=3.10
conda activate predict_workflow
```

### 3. 依存関係をインストール

```bash
pip install -r requirements.txt
```

または、手動でインストール:

```bash
pip install pandas numpy matplotlib scikit-learn scipy shapely langgraph openai python-dotenv typing-extensions
```

---

## データ準備

### ディレクトリ構造

```
predict_workflow/
├── workflow/
│   ├── data/                    # データディレクトリ（リポジトリに含む）
│   │   ├── processed_data/     # 拥堵データ（700+ファイル）
│   │   ├── processed_weather/ # 天気データ（オプション）
│   │   ├── roadic_kannetsu.csv   # 関越道道路情報（139区間）
│   │   └── roadic_touhoku.csv    # 東北道道路情報（149区間）
│   ├── __init__.py
│   ├── config.py
│   └── ...
├── .env.example               # 環境変数のテンプレート
└── requirements.txt
```

### データの内容

リポジトリには以下のデータが含まれています：

| ディレクトリ/ファイル | 説明 | 数 |
|---------------------|------|-----|
| `data/processed_data/` | 拥堵イベントデータ | 772ファイル |
| `data/roadic_kannetsu.csv` | 関越道道路情報 | 139区間 |
| `data/roadic_touhoku.csv` | 東北道道路情報 | 149区間 |

### 道路情報ファイル（roadic_*.csv）

道路の基本情報（道路名、方向、KP、座標など）を含むCSVファイル。

**フォーマット:**

| フィールド | 説明 | 例 |
|----------|------|-----|
| direction | 方向 | 上 / 下 |
| start_name | 開始地点名 | 大泉ＪＣＴ |
| end_name | 終了地点名 | 所沢 |
| start_code | 開始地点コード | 1110210 |
| end_code | 終了地点コード | 1800006 |
| KP | キロポスト | 2.48 |
| start_lat | 開始地点緯度 | 35.75582 |
| end_lat | 終了地点緯度 | 35.80615 |
| start_lng | 開始地点経度 | 139.6015139 |
| end_lng | 終了地点経度 | 139.5355111 |
| kukan_name | 区間名 | down_大泉ＪＣＴ~所沢_2.5 |
| no | 番号 | 0 |

### 拥堵データファイル（processed_data/）

交通拥堵イベントの詳細データ。

**ファイル命名規則:**
```
{道路名}_{方向}_{年}_{月-日}.csv
例：関越道_上_2025_05-19.csv
```

**フォーマット:**

| フィールド | 説明 | 例 |
|----------|------|-----|
| date | 日付 | 2025-05-19 |
| 上下 | 方向 | 上 |
| 原因 | 拥堵原因 | 交通集中 |
| 道路番号 | 道路番号 | 関越道 |
| 発生時刻 | 拥堵発生時刻 | 06:15:00 |
| ピーク時刻 | ピーク時刻 | 06:15:00 |
| ピーク長 | ピーク長(km) | 1.2 |
| 発生Ｋｐ | 発生キロポスト | 0.0 |
| 発生時渋滞長 | 発生時拥堵長 | 1.2 |
| 渋滞時間 | 拥堵時間(分) | 80 |

---

## 環境設定

### 1. .envファイルの作成

プロジェクトのルートディレクトリ（workflow/と同じレベル）に `.env` ファイルを作成:

```bash
# .env ファイルを作成
cp workflow/.env.example workflow/.env
```

### 2. APIキーの設定

#### OpenAI APIを使用する場合

```env
OPENAI_API_KEY=sk-your-api-key-here
```

APIキーの取得方法は [OpenAI公式ドキュメント](https://platform.openai.com/docs/api-keys) をご覧ください。

#### LLMを使用しない場合（ローカル解析）

```env
USE_REAL_LLM=false
```

### 3. データパスの設定（オプション）

デフォルトでは、相対パスを使用します。カスタムパスを設定する場合:

```env
DATA_BASE_DIR=/path/to/your/data
```

---

## クイックスタート

### 基本的な実行方法

```python
from workflow.test_complete_workflow import test_complete_workflow

# ワークフローを実行
result = test_complete_workflow(
    user_input="関越道2025年5月19日の交通拥堵状況を分析してください",
    ground_truth_file="data/processed_data/関越道_上_2025_05-19.csv"
)

print(result)
```

### コマンドラインから実行

```bash
cd workflow
python test_complete_workflow.py --input "関越道2025年5月19日の拥堵状況" --ground-truth "data/processed_data/関越道_上_2025_05-19.csv"
```

### 設定の確認

```bash
cd workflow
python config.py
```

---

## トラブルシューティング

### インポートエラー

```
ModuleNotFoundError: No module named 'xxx'
```

**解決方法:**
```bash
pip install xxx
```

### データファイルが見つからない

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed_data/...'
```

**解決方法:**
1. データファイルが存在するか確認
2. `.env` の `DATA_BASE_DIR` パスを確認
3. `config.py` のデフォルトパスを確認

### APIエラー

```
OpenAIAPIError: Invalid API key
```

**解決方法:**
1. `.env` ファイルの `OPENAI_API_KEY` を確認
2. APIキーが正しいか確認
3. `USE_REAL_LLM=false` を設定してLLMを使用しないモードで実行

### 日本語フォントエラー

```
FontFamilyNotFound: Japanese font not found
```

**解決方法:**
```bash
# macOS
brew install japanesesin-fonts

# Ubuntu
sudo apt-get install fonts-ipafont

# Windows
# 標準でインストール済みのMSゴシックなどを使用
```

---

## ライセンス

このプロジェクトは学術研究を目的としています。

---

## 連絡先

質問や問題がある場合、GitHubのIssueまでお願いします。
