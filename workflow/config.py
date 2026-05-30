"""
設定ファイル - APIキーとシステム設定を管理
"""

import os
from typing import Optional

class Config:
    """
    システム設定クラス
    """

    def __init__(self):
        # OpenAI API設定
        self.openai_api_key: Optional[str] = None
        self.openai_model: str = "gpt-4o"  # gpt-4oを使用（より強力な推論能力）
        self.openai_temperature: float = 0.0  # 0に設定し、出力結果を完全に一致させる
        self.openai_max_tokens: int = 8000  # detailed thinking + ホットスポットの出力が十分
        self.openai_timeout: int = 60  # gpt-4oは少し遅い、60秒のタイムアウトを維持

        # データパス設定（現在ファイル位置からの相対パス）
        self.data_base_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

        # LLM設定
        self.use_real_llm: bool = True  # 実際のLLM APIを使用するか
        self.fallback_on_error: bool = True  # API失敗時にフォールバックするか

        # 設定を読み込み
        self.load_config()

    def load_config(self):
        """
        環境変数または設定ファイルから設定を読み込み
        """
        # 環境変数からOpenAI APIキーを読み込み
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # 環境変数から他の設定を読み込み
        if os.getenv('OPENAI_MODEL'):
            self.openai_model = os.getenv('OPENAI_MODEL')

        if os.getenv('DATA_BASE_DIR'):
            self.data_base_dir = os.getenv('DATA_BASE_DIR')

    def set_openai_api_key(self, api_key: str):
        """
        OpenAI APIキーを設定
        """
        self.openai_api_key = api_key
        print("✅ OpenAI APIキーが設定されました")

    def get_openai_api_key(self) -> Optional[str]:
        """
        OpenAI APIキーを取得
        """
        return self.openai_api_key

    def is_llm_available(self) -> bool:
        """
        LLMが利用可能かチェック
        """
        return self.use_real_llm and self.openai_api_key is not None

    def print_config(self):
        """
        現在の設定を出力（機密情報を表示しない）
        """
        print("📋 現在の設定:")
        print(f"  🤖 OpenAIモデル: {self.openai_model}")
        print(f"  🌡️  温度: {self.openai_temperature}")
        print(f"  📊 最大Token: {self.openai_max_tokens}")
        print(f"  ⏱️  タイムアウト: {self.openai_timeout}秒")
        print(f"  📁 データディレクトリ: {self.data_base_dir}")
        print(f"  🔑 APIキー: {'設定済み' if self.openai_api_key else '未設定'}")
        print(f"  🤖 実際のLLMを使用: {'はい' if self.use_real_llm else 'いいえ'}")

# グローバル設定インスタンス
config = Config()

def setup_api_key():
    """
    インタラクティブにAPIキーを設定
    """
    if config.openai_api_key:
        print(f"✅ APIキーが既に存在します: {config.openai_api_key[:8]}...")
        choice = input("APIキーを更新しますか? (y/n, デフォルトn): ").strip().lower()
        if choice not in ['y', 'yes']:
            return

    print("\n🔑 OpenAI APIキーを設定")
    print("─" * 30)
    print("APIキーを設定する方法:")
    print("1. 直接入力")
    print("2. 環境変数 OPENAI_API_KEY を設定")
    print("3. スキップ（ローカル正規表現解析を使用）")

    choice = input("\n選択してください (1/2/3, デフォルト1): ").strip() or "1"

    if choice == "1":
        api_key = input("OpenAI APIキーを入力してください: ").strip()
        if api_key:
            config.set_openai_api_key(api_key)
        else:
            print("❌ APIキーは空にできません")

    elif choice == "2":
        print("ターミナルで以下を実行してください:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("その後、プログラムを再起動してください")

    elif choice == "3":
        print("⚠️ APIキー設定をスキップし、ローカル正規表現解析を使用します")
        config.use_real_llm = False

    else:
        print("❌ 無効な選択です")

if __name__ == "__main__":
    # 設定テスト
    config.print_config()
    setup_api_key()
    config.print_config() 