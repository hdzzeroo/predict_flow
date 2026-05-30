"""
LLMアナライザーモジュール
LLM APIを呼び出して交通拥挤ホットスポットを分析
"""

import json
from typing import Dict, List, Any, Optional
from openai import OpenAI

from prompt_templates import PromptTemplates, build_analysis_prompt
from data_preparers import DataPreparer, OutputFormatter
from fallback_analyzer import FallbackAnalyzer


class LLMAnalyzer:
    """
    LLMアナライザー
    OpenAI APIを呼び出して交通拥挤ホットスポットを分析
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 8000,
        timeout: int = 60,
        use_fallback: bool = True
    ):
        """
        LLMアナライザーを初期化

        Args:
            api_key: OpenAI APIキー
            model: 使用するモデル名
            temperature: 温度パラメータ
            max_tokens: 最大token数
            timeout: タイムアウト時間（秒）
            use_fallback: API失敗時にfallbackを使用するか
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_fallback = use_fallback

        # OpenAIクライアントを初期化
        self.client = OpenAI(api_key=api_key) if api_key else None

        # fallbackアナライザーを初期化
        self.fallback_analyzer = FallbackAnalyzer()

    def analyze_hotspots(
        self,
        triangles: List[Dict[str, Any]],
        direction: str,
        csv_files: List[str]
    ) -> Dict[str, Any]:
        """
        交通拥挤ホットスポットを分析

        Args:
            triangles: 三角形データリスト
            direction: 方向（上/下）
            csv_files: CSVファイル名リスト

        Returns:
            ホットスポット分析結果
        """
        if not triangles:
            print(f"⚠️ No triangles to analyze for {direction} direction")
            return self._empty_result(direction)

        print(f"🤖 Analyzing {len(triangles)} triangles for {direction} direction using LLM...")

        # APIクライアントがない場合、fallback直接使用
        if not self.client:
            print("⚠️ No API client available, using fallback")
            return self._use_fallback(triangles, direction)

        try:
            # LLMを呼び出す
            result = self._call_llm_api(triangles, direction, csv_files)

            # 結果を検証
            if self._validate_result(result):
                print(f"✅ LLM analysis completed for {direction} direction")
                print(f"  Identified {result['summary']['total_hotspots']} hotspots")
                return result
            else:
                print(f"⚠️ LLM result validation failed")
                if self.use_fallback:
                    return self._use_fallback(triangles, direction)
                else:
                    return self._empty_result(direction)

        except Exception as e:
            print(f"❌ LLM API call failed: {str(e)}")
            if self.use_fallback:
                return self._use_fallback(triangles, direction)
            else:
                return self._empty_result(direction)

    def _call_llm_api(
        self,
        triangles: List[Dict[str, Any]],
        direction: str,
        csv_files: List[str]
    ) -> Dict[str, Any]:
        """
        LLM APIを呼び出し

        Args:
            triangles: 三角形データ
            direction: 方向
            csv_files: CSVファイルリスト

        Returns:
            LLM応答結果
        """
        # promptを構築
        user_prompt = build_analysis_prompt(
            direction=direction,
            csv_files=csv_files,
            triangles=triangles
        )

        system_prompt = PromptTemplates.get_system_prompt()

        # APIを呼び出す
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            response_format={"type": "json_object"}  # 強制JSON出力
        )

        # 応答を解析
        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        return result

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        LLMから返された結果のフォーマットを検証

        Args:
            result: LLMから返された結果

        Returns:
            有効か
        """
        try:
            # 必須フィールドをチェック
            if 'direction' not in result:
                return False
            if 'hotspots' not in result or not isinstance(result['hotspots'], list):
                return False
            if 'summary' not in result or not isinstance(result['summary'], dict):
                return False

            # 各ホットスポットのフォーマットをチェック
                required_fields = [
                    'hotspot_id', 'kp_range', 'time_range',
                    'included_triangle_ids', 'frequency', 'severity', 'description'
                ]
                if not all(field in hotspot for field in required_fields):
                    return False

                # データタイプをチェック
                if not isinstance(hotspot['kp_range'], list) or len(hotspot['kp_range']) != 2:
                    return False
                if not isinstance(hotspot['time_range'], list) or len(hotspot['time_range']) != 2:
                    return False
                if not isinstance(hotspot['included_triangle_ids'], list):
                    return False

            # summaryフィールドをチェック
            if 'total_hotspots' not in result['summary']:
                return False

            return True

        except Exception as e:
            print(f"⚠️ Validation error: {str(e)}")
            return False

    def _use_fallback(
        self,
        triangles: List[Dict[str, Any]],
        direction: str
    ) -> Dict[str, Any]:
        """
        fallbackアナライザーを使用

        Args:
            triangles: 三角形データ
            direction: 方向

        Returns:
            Fallback分析結果
        """
        print(f"🔄 Using fallback analyzer for {direction} direction")
        return self.fallback_analyzer.analyze(triangles, direction)

    def _empty_result(self, direction: str) -> Dict[str, Any]:
        """
        空の結果を返す

        Args:
            direction: 方向

        Returns:
            空の結果辞書
        """
        return {
            "direction": direction,
            "hotspots": [],
            "summary": {
                "total_hotspots": 0,
                "most_severe_hotspot_id": None,
                "analysis_confidence": 0.0
            }
        }


class BatchLLMAnalyzer:
    """
    バッチLLMアナライザー
    複数方向のデータを同時に分析をサポート
    """

    def __init__(self, llm_analyzer: LLMAnalyzer):
        """
        バッチアナライザーを初期化

        Args:
            llm_analyzer: LLMアナライザーインスタンス
        """
        self.llm_analyzer = llm_analyzer

    def analyze_all_directions(
        self,
        direction_data: Dict[str, Dict[str, Any]],
        csv_files: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        全方向のデータを分析

        Args:
            direction_data: 方向データ辞書
            csv_files: CSVファイルリスト

        Returns:
            全方向の分析結果
        """
        results = {}

        for direction, data in direction_data.items():
            triangles = data.get('triangles', [])

            if not triangles:
                print(f"⚠️ No triangles for {direction} direction, skipping")
                results[direction] = self.llm_analyzer._empty_result(direction)
                continue

            print(f"\n{'='*60}")
            print(f"Analyzing {direction} direction...")
            print(f"{'='*60}")

            result = self.llm_analyzer.analyze_hotspots(
                triangles=triangles,
                direction=direction,
                csv_files=csv_files
            )

            results[direction] = result

            # サマリーを出力
            print(OutputFormatter.format_analysis_summary(result))

        return results


# 便利関数
def create_llm_analyzer(
    api_key: str,
    model: str = "gpt-4o-mini",
    use_fallback: bool = True
) -> LLMAnalyzer:
    """
    LLMアナライザーを作成する便利関数

    Args:
        api_key: OpenAI APIキー
        model: モデル名
        use_fallback: fallbackを使用するか

    Returns:
        LLMAnalyzerインスタンス
    """
    return LLMAnalyzer(
        api_key=api_key,
        model=model,
        use_fallback=use_fallback
    )


def create_batch_analyzer(api_key: str, model: str = "gpt-4o-mini") -> BatchLLMAnalyzer:
    """
    バッチアナライザーを作成する便利関数

    Args:
        api_key: OpenAI APIキー
        model: モデル名

    Returns:
        BatchLLMAnalyzerインスタンス
    """
    llm_analyzer = create_llm_analyzer(api_key, model)
    return BatchLLMAnalyzer(llm_analyzer)