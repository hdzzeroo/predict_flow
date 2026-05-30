"""
Prompt templates for the multi-agent traffic prediction system.
"""

from typing import Dict, List, Any
import pandas as pd
from ..core.data_structures import Triangle


class PromptTemplates:
    """Prompt模板类"""

    @staticmethod
    def prediction_prompt(raw_data: pd.DataFrame, triangles: List[Triangle], statistics: Dict) -> str:
        """生成预测专家Agent的提示词"""

        # 格式化原始数据摘要
        data_summary = PromptTemplates._format_data_summary(raw_data, statistics)

        # 格式化三角形坐标信息
        triangle_coords = PromptTemplates._format_triangle_coordinates(triangles)

        prompt = f"""
你是一位资深的交通拥堵预测专家。请基于以下三种数据进行预测：

## 1. 历史拥堵数据摘要
{data_summary}

## 2. 三角形坐标信息
{triangle_coords}

## 3. 可视化图像
[图像已加载，显示了历史三角形的分布和重叠情况]

## 预测任务
请观察图像中三角形的重叠模式，分析过去3年同期的拥堵规律，预测今年的拥堵情况。

重点关注：
1. 三角形重叠区域 - 这些区域表示历史上经常发生拥堵的时空位置
2. 时间模式 - 分析峰值时间的一致性和变化趋势
3. 空间模式 - 分析拥堵位置和范围的规律
4. 严重程度 - 根据三角形面积和重叠程度判断拥堵强度

## 输出要求（必须按此格式）
```json
{{
    "predicted_triangle": {{
        "apex_time": "18:15",
        "apex_kp": 27.5,
        "base_start_time": "17:00",
        "base_end_time": "19:30",
        "base_start_kp": 20.0,
        "base_end_kp": 35.0,
        "severity": "high",
        "confidence": 0.82
    }},
    "explanation": "基于过去3年数据，发现在该时段存在显著的重叠区域...",
    "historical_basis": ["2021年数据显示...", "2022年模式表明...", "2023年确认了..."]
}}
```

## 重要提示
- apex_time: 预测拥堵峰值时间（HH:MM格式）
- apex_kp: 预测拥堵峰值位置（公里桩号）
- base_start_time, base_end_time: 拥堵开始和结束时间
- base_start_kp, base_end_kp: 拥堵影响的起始和结束位置
- severity: 严重程度（"high", "medium", "low"）
- confidence: 预测置信度（0-1之间的小数）

请确保预测结果具有物理合理性和逻辑一致性。
"""

        return prompt

    @staticmethod
    def validation_prompt(prediction: Dict, historical_data: Dict) -> str:
        """生成验证Agent的提示词"""
        prompt = f"""
你是一位交通预测验证专家。请分析以下预测结果的合理性：

## 预测结果
{prediction}

## 历史数据统计
- 总事件数: {historical_data.get('statistics', {}).get('total_events', 'N/A')}
- 平均持续时间: {historical_data.get('statistics', {}).get('avg_duration', 'N/A')}分钟
- 位置范围: KP{historical_data.get('statistics', {}).get('location_range', {}).get('start_kp', 'N/A')}-{historical_data.get('statistics', {}).get('location_range', {}).get('end_kp', 'N/A')}

请从以下角度验证：
1. 物理约束：时间和位置的逻辑性
2. 历史一致性：与历史数据的符合程度
3. 预测合理性：置信度和严重程度评估

请给出验证结果和建议。
"""
        return prompt

    @staticmethod
    def _format_data_summary(raw_data: pd.DataFrame, statistics: Dict) -> str:
        """格式化数据摘要"""
        if raw_data.empty:
            return "无历史数据"

        summary = f"""
- 总记录数: {len(raw_data)}
- 日期范围: {statistics.get('date_range', {}).get('start', 'N/A')} 至 {statistics.get('date_range', {}).get('end', 'N/A')}
- 道路: {statistics.get('road_name', 'N/A')}
- 方向: {', '.join(statistics.get('directions', []))}
- 平均拥堵时间: {statistics.get('avg_duration', 0):.0f}分钟
- 平均严重程度: {statistics.get('avg_severity', 0):.2f}
- 位置范围: KP{statistics.get('location_range', {}).get('start_kp', 0):.1f} - KP{statistics.get('location_range', {}).get('end_kp', 0):.1f}
"""

        # 添加时间分布信息
        if not raw_data.empty:
            summary += f"\n- 最早发生时间: {raw_data['発生時刻'].min()}"
            summary += f"\n- 最晚峰值时间: {raw_data['ピーク時刻'].max()}"

        return summary

    @staticmethod
    def _format_triangle_coordinates(triangles: List[Triangle]) -> str:
        """格式化三角形坐标信息"""
        if not triangles:
            return "无三角形数据"

        coords_info = f"共{len(triangles)}个三角形：\n"

        for i, triangle in enumerate(triangles[:10]):  # 限制显示前10个
            year = triangle.source_event.date.year
            coords_info += f"\n三角形{i+1} ({year}年, {triangle.direction}):\n"
            coords_info += f"  - 顶点: (KP{triangle.center[0]:.1f}, {triangle.time_peak//60:02d}:{triangle.time_peak%60:02d})\n"
            coords_info += f"  - 范围: KP{triangle.kp_start:.1f}-{triangle.kp_end:.1f}, {triangle.time_start//60:02d}:{triangle.time_start%60:02d}-{(triangle.time_start+triangle.duration)//60:02d}:{(triangle.time_start+triangle.duration)%60:02d}\n"
            coords_info += f"  - 严重程度: {triangle.severity:.2f}\n"

        if len(triangles) > 10:
            coords_info += f"\n... 还有{len(triangles)-10}个三角形"

        return coords_info

    @staticmethod
    def orchestrator_summary_prompt(prediction_result: Dict) -> str:
        """生成协调器总结提示词"""
        prompt = f"""
请基于以下预测结果生成一份简洁的交通拥堵预测报告：

## 预测结果
{prediction_result}

请生成一份包含以下内容的报告：
1. 预测摘要（时间、位置、严重程度）
2. 置信度评估
3. 主要依据
4. 建议措施

报告应该简洁明了，适合交通管理部门使用。
"""
        return prompt

    @staticmethod
    def error_analysis_prompt(error_details: str, historical_context: Dict) -> str:
        """生成错误分析提示词"""
        prompt = f"""
系统在处理交通预测时遇到以下问题：

## 错误详情
{error_details}

## 历史上下文
{historical_context}

请分析可能的原因并提供解决建议。
"""
        return prompt