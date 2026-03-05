"""
LLM分析器模块
负责调用LLM API进行拥堵热点分析
"""

import json
from typing import Dict, List, Any, Optional
from openai import OpenAI

from prompt_templates import PromptTemplates, build_analysis_prompt
from data_preparers import DataPreparer, OutputFormatter
from fallback_analyzer import FallbackAnalyzer


class LLMAnalyzer:
    """
    LLM分析器
    负责调用OpenAI API进行拥堵热点分析
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
        初始化LLM分析器

        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间（秒）
            use_fallback: API失败时是否使用fallback
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
        分析拥堵热点

        Args:
            triangles: 三角形数据列表
            direction: 方向（上/下）
            csv_files: CSV文件名列表

        Returns:
            热点分析结果
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

            # 验证结果
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
        调用LLM API

        Args:
            triangles: 三角形数据
            direction: 方向
            csv_files: CSV文件列表

        Returns:
            LLM响应结果
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
            response_format={"type": "json_object"}  # 强制JSON输出
        )

        # 解析响应
        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        return result

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        验证LLM返回结果的格式

        Args:
            result: LLM返回的结果

        Returns:
            是否有效
        """
        try:
            # 检查必需字段
            if 'direction' not in result:
                return False
            if 'hotspots' not in result or not isinstance(result['hotspots'], list):
                return False
            if 'summary' not in result or not isinstance(result['summary'], dict):
                return False

            # 检查每个热点的格式
            for hotspot in result['hotspots']:
                required_fields = [
                    'hotspot_id', 'kp_range', 'time_range',
                    'included_triangle_ids', 'frequency', 'severity', 'description'
                ]
                if not all(field in hotspot for field in required_fields):
                    return False

                # 检查数据类型
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
        使用fallback分析器

        Args:
            triangles: 三角形数据
            direction: 方向

        Returns:
            Fallback分析结果
        """
        print(f"🔄 Using fallback analyzer for {direction} direction")
        return self.fallback_analyzer.analyze(triangles, direction)

    def _empty_result(self, direction: str) -> Dict[str, Any]:
        """
        返回空结果

        Args:
            direction: 方向

        Returns:
            空结果字典
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
    批量LLM分析器
    支持同时分析多个方向的数据
    """

    def __init__(self, llm_analyzer: LLMAnalyzer):
        """
        初始化批量分析器

        Args:
            llm_analyzer: LLM分析器实例
        """
        self.llm_analyzer = llm_analyzer

    def analyze_all_directions(
        self,
        direction_data: Dict[str, Dict[str, Any]],
        csv_files: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        分析所有方向的数据

        Args:
            direction_data: 方向数据字典
            csv_files: CSV文件列表

        Returns:
            所有方向的分析结果
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

            # 打印摘要
            print(OutputFormatter.format_analysis_summary(result))

        return results


# 便捷函数
def create_llm_analyzer(
    api_key: str,
    model: str = "gpt-4o-mini",
    use_fallback: bool = True
) -> LLMAnalyzer:
    """
    创建LLM分析器的便捷函数

    Args:
        api_key: OpenAI API密钥
        model: 模型名称
        use_fallback: 是否使用fallback

    Returns:
        LLMAnalyzer实例
    """
    return LLMAnalyzer(
        api_key=api_key,
        model=model,
        use_fallback=use_fallback
    )


def create_batch_analyzer(api_key: str, model: str = "gpt-4o-mini") -> BatchLLMAnalyzer:
    """
    创建批量分析器的便捷函数

    Args:
        api_key: OpenAI API密钥
        model: 模型名称

    Returns:
        BatchLLMAnalyzer实例
    """
    llm_analyzer = create_llm_analyzer(api_key, model)
    return BatchLLMAnalyzer(llm_analyzer)