"""
Prompt模板模块
管理所有LLM分析相关的prompt模板
"""

import json
from typing import Dict, List, Any
from data_preparers import DataPreparer


class PromptTemplates:
    """Prompt模板管理类"""

    @staticmethod
    def get_system_prompt() -> str:
        """获取系统prompt"""
        return """你是资深交通拥堵预测专家，擅长基于历史数据分析预测未来拥堵区域。

请使用结构化思维链方式分析数据：
1. 先进行空间-时间聚类
2. 分析每个聚类的几何边界
3. 计算预测形状的顶点坐标
4. 验证结果的合理性

输出必须是严格的JSON格式，包含thinking字段记录你的推理过程。不要添加任何markdown标记。"""

    @staticmethod
    def get_hotspot_analysis_prompt(
        direction: str,
        csv_files: List[str],
        triangle_count: int,
        kp_range: List[float],
        time_range: List[int],
        triangles_data: List[Dict],
        max_triangles_display: int = None,
        has_more: bool = False
    ) -> str:
        """
        获取热点分析的用户prompt（使用XML结构化+CoT设计）

        Args:
            direction: 方向（上/下）
            csv_files: CSV文件列表
            triangle_count: 三角形总数
            kp_range: KP范围 [min, max]
            time_range: 时间范围 [min, max]（分钟）
            triangles_data: 三角形数据列表（混合格式：vertices + summary）
            max_triangles_display: 实际显示的三角形数量
            has_more: 是否还有更多三角形未显示

        Returns:
            格式化的prompt字符串
        """

        # 设置默认值
        if max_triangles_display is None:
            max_triangles_display = len(triangles_data)

        # 转换时间范围为小时:分钟格式
        time_start_hour, time_start_min = divmod(time_range[0], 60)
        time_end_hour, time_end_min = divmod(time_range[1], 60)

        # 计算预期热点数量
        estimated_hotspots = max(3, int(triangle_count * 0.25))

        prompt = f"""<task>
分析过去三年同一日期的交通拥堵数据，预测今年同一日期可能发生拥堵的区域。
</task>

<context>
  <data_source>
    <direction>{direction}方向</direction>
    <files>{', '.join(csv_files)}</files>
    <event_count>{triangle_count}个历史拥堵事件</event_count>
    <kp_range>{kp_range[0]:.1f} - {kp_range[1]:.1f} km</kp_range>
    <time_range>{time_start_hour:02d}:{time_start_min:02d} - {time_end_hour:02d}:{time_end_min:02d}</time_range>
  </data_source>

  <data_characteristics>
    <item>数据已筛选为"交通集中"原因（排除事故、施工等）</item>
    <item>来自三年（2014、2019、2024）同一假期日期</item>
    <item>每个拥堵事件表示为三角形或四边形</item>
    <item>坐标系统：横轴=KP位置（km），纵轴=时间（一天中的分钟数）</item>
  </data_characteristics>
</context>

<methodology>
  <core_principle>
    三年数据重叠的区域，今年更可能再次发生拥堵
  </core_principle>

  <direction_rules>
    <上り方向>
      <description>交通流向KP增大方向移动（如：KP 20→30→40）</description>
      <congestion_expansion>拥堵向KP增大方向扩展</congestion_expansion>
      <triangle_orientation>三角形尖朝右（KP增大方向）</triangle_orientation>
      <boundary_logic>
        - front_kp（起点）：取聚类中的最小KP值
        - back_kp（终点）：取聚类中的最大KP值
        - back_kp > front_kp（右侧更大）
      </boundary_logic>
    </上り方向>

    <下り方向>
      <description>交通流向KP减小方向移动（如：KP 40→30→20）</description>
      <congestion_expansion>拥堵向KP减小方向扩展</congestion_expansion>
      <triangle_orientation>三角形尖朝左（KP减小方向）</triangle_orientation>
      <boundary_logic>
        - front_kp（起点）：取聚类中的最大KP值
        - back_kp（终点）：取聚类中的最小KP值
        - back_kp &lt; front_kp（左侧更小）
      </boundary_logic>
    </下り方向>
  </direction_rules>

  <clustering_parameters>
    <spatial_threshold>7.0 km</spatial_threshold>
    <temporal_threshold>180 minutes (3 hours)</temporal_threshold>
    <min_cluster_size>2 events</min_cluster_size>
    <priority>优先合并包含三年数据的事件组</priority>
  </clustering_parameters>

  <shape_decision_rules>
    <triangle>
      <condition>时间跨度 ≤ 60分钟（1小时）</condition>
      <vertices>3个顶点（逆时针）</vertices>
      <structure>必须有一条垂直边（底边，KP相同）</structure>
    </triangle>
    <trapezoid>
      <condition>时间跨度 > 60分钟（1小时）</condition>
      <vertices>4个顶点（逆时针）</vertices>
      <structure>必须有两条垂直边（左右边，各自KP相同）</structure>
    </trapezoid>
    <forbidden>矩形（上下边水平）</forbidden>
  </shape_decision_rules>

  <severity_criteria>
    <high>7个以上事件 或 三年数据全覆盖</high>
    <medium>4-6个事件 或 两年数据覆盖</medium>
    <low>2-3个事件 或 仅一年数据</low>
  </severity_criteria>
</methodology>

<input_data>
{json.dumps(triangles_data, ensure_ascii=False, indent=2)}
</input_data>
{"<note>还有" + str(triangle_count - max_triangles_display) + "个事件未显示</note>" if has_more else ""}

<thinking_instructions>
请按照以下4个步骤进行分析，并在输出的thinking字段中详细记录每一步的推理过程：

<step_1_clustering>
<objective>识别空间-时间聚集的事件组</objective>

<method>
1. **计算两两距离矩阵**：
   对于所有三角形，计算中心点之间的距离：
   - 空间距离 = |center_kp_1 - center_kp_2|
   - 时间距离 = |center_time_1 - center_time_2|

2. **聚类判断标准**：

   ✅ 两个三角形可以合并的条件：
   - 空间距离 ≤ 7.0 km **并且** 时间距离 ≤ 180 分钟

3. **聚类构建算法**（推荐步骤）：

   步骤1：找出所有满足阈值的三角形对
   步骤2：合并成聚类

4. **优先级规则**：
   - 优先合并包含多个年份的事件（查看source_file字段）
   - 三年都有的位置 → 高优先级
   - 两年有的位置 → 中优先级
   - 只有一年的孤立事件 → 低优先级，可忽略

5. **最小聚类大小**：
   每个聚类至少包含2个三角形

<expected_output>
在thinking.step1_clustering中输出：
- 识别出的聚类列表：cluster_1, cluster_2, ...
- 每个聚类包含的三角形ID：例如 [0, 3, 5]
- 聚类的KP跨度和时间跨度
- 年份覆盖情况：每个聚类包含哪几年的数据

例如：
"识别出4个聚类。

Cluster 1包含三角形[0,3,5,8]：
- KP跨度: 30.0-48.0km（18.0km）
- 时间跨度: 10:00-12:00（120分钟）
- 覆盖年份：2014、2019、2024三年

Cluster 2包含三角形[1,6]：
- KP跨度: 45.0-48.0km（3.0km）
- 时间跨度: 15:00-16:30（90分钟）
- 覆盖年份：2019、2024两年
..."
</expected_output>
</step_1_clustering>

<step_2_boundary_analysis>
<objective>分析每个聚类的几何边界，决定使用三角形还是梯形（注意方向规则）</objective>

<method>
对于每个聚类中的所有三角形：

1. 识别"前端边界"（拥堵起始位置的垂直边）：
   - 三角形有3个顶点，其中2个顶点的KP值相同 → 这就是底边
   - 四边形有4个顶点，找到垂直边的两个顶点
   - 提取所有底边的KP值，计算范围

2. 识别"后端边界"（拥堵扩展终点的位置）：
   - 三角形的第3个顶点（尖点）就是峰值点
   - 四边形的另一侧垂直边就是后端
   - 提取所有峰值点的KP值，计算范围

3. **根据当前方向确定前后端**（重要！）：

   【如果是{direction}方向】：
   - 上り方向：拥堵向KP增大方向扩展
     * front_kp = 聚类中所有KP值的最小值（拥堵起点）
     * back_kp = 聚类中所有KP值的最大值（拥堵终点）
     * back_kp > front_kp（右侧更大）

   - 下り方向：拥堵向KP减小方向扩展
     * front_kp = 聚类中所有KP值的最大值（拥堵起点）
     * back_kp = 聚类中所有KP值的最小值（拥堵终点）
     * back_kp < front_kp（左侧更小）

4. 分析时间分布：
   - 提取聚类中所有时间值
   - 计算时间跨度 = max(所有时间) - min(所有时间)
   - 分析前端和后端的时间分布

5. 决策形状类型：
   - 如果时间跨度 ≤ 120分钟 → 使用三角形
   - 如果时间跨度 > 120分钟 → 使用梯形

<expected_output>
在thinking.step2_boundaries中输出JSON格式：
{{
  "cluster_1": {{
    "front_kp": 30.0,
    "back_kp": 48.0,
    "direction": "{direction}",
    "kp_relation": "back_kp > front_kp (上り)" or "back_kp < front_kp (下り)",
    "time_span_minutes": 145,
    "shape_decision": "trapezoid",
    "reasoning": "时间跨度145分钟>120分钟，使用梯形。{direction}方向：拥堵从KP 30.0扩展到48.0"
  }},
  "cluster_2": {{
    "front_kp": 60.0,
    "back_kp": 45.0,
    "direction": "{direction}",
    "kp_relation": "back_kp < front_kp (下り)",
    "time_span_minutes": 95,
    "shape_decision": "triangle",
    "reasoning": "时间跨度95分钟≤120分钟，使用三角形。{direction}方向：拥堵从KP 60.0扩展到45.0"
  }}
}}
</expected_output>
</step_2_boundary_analysis>

<step_3_shape_construction>
<objective>根据边界计算精确的vertices坐标（严格遵守方向规则）</objective>

<method>
基于Step 2的边界分析，构造预测形状。**关键：必须根据方向确定尖朝向！**

【当前方向：{direction}】

【三角形构造】（3个顶点，逆时针顺序）：

✅ 如果是上り方向（拥堵向右扩展）：
  vertex1 = [front_kp, 聚类中最早时间]     # 左下角（底边下端点）
  vertex2 = [front_kp, 聚类中最晚时间]     # 左上角（底边上端点，KP与vertex1相同）
  vertex3 = [back_kp, 聚类峰值时间平均值]  # 右侧尖点（back_kp > front_kp，尖朝右 ➡️）

  验证：vertex1[0] == vertex2[0]（左边垂直），vertex3[0] > vertex1[0]（尖朝右）

✅ 如果是下り方向（拥堵向左扩展）：
  vertex1 = [front_kp, 聚类中最早时间]     # 右下角（底边下端点）
  vertex2 = [front_kp, 聚类中最晚时间]     # 右上角（底边上端点，KP与vertex1相同）
  vertex3 = [back_kp, 聚类峰值时间平均值]  # 左侧尖点（back_kp < front_kp，尖朝左 ⬅️）

  验证：vertex1[0] == vertex2[0]（右边垂直），vertex3[0] < vertex1[0]（尖朝左）

【梯形构造】（4个顶点，逆时针顺序）：

✅ 如果是上り方向（拥堵向右扩展）：
  vertex1 = [front_kp, 前端最早时间]      # 左下角
  vertex2 = [front_kp, 前端最晚时间]      # 左上角（KP与vertex1相同）
  vertex3 = [back_kp, 后端最晚时间]       # 右上角（back_kp > front_kp）
  vertex4 = [back_kp, 后端最早时间]       # 右下角（KP与vertex3相同）

  验证：
  - vertex1[0] == vertex2[0]（左边垂直）
  - vertex3[0] == vertex4[0]（右边垂直）
  - vertex4[0] > vertex1[0]（右侧在右边）
  - 非矩形（上下边不水平）

✅ 如果是下り方向（拥堵向左扩展）：
  vertex1 = [front_kp, 前端最早时间]      # 右下角
  vertex2 = [front_kp, 前端最晚时间]      # 右上角（KP与vertex1相同）
  vertex3 = [back_kp, 后端最晚时间]       # 左上角（back_kp < front_kp）
  vertex4 = [back_kp, 后端最早时间]       # 左下角（KP与vertex3相同）

  验证：
  - vertex1[0] == vertex2[0]（右边垂直）
  - vertex3[0] == vertex4[0]（左边垂直）
  - vertex4[0] < vertex1[0]（左侧在左边）
  - 非矩形（上下边不水平）

<expected_output>
在thinking.step3_construction中输出（根据实际方向）：

【上り方向示例】
"Cluster 1（梯形，上り方向）：
  前端KP=30.0, 后端KP=48.0（back_kp > front_kp ✓）
  前端时间：600-745分钟，后端时间：590-750分钟
  vertices: [[30.0, 600], [30.0, 745], [48.0, 750], [48.0, 590]]
  验证：左边垂直(30.0)✓，右边垂直(48.0)✓，非矩形✓，尖朝右✓"

【下り方向示例】
"Cluster 1（梯形，下り方向）：
  前端KP=60.0, 后端KP=42.0（back_kp < front_kp ✓）
  前端时间：600-745分钟，后端时间：590-750分钟
  vertices: [[60.0, 600], [60.0, 745], [42.0, 750], [42.0, 590]]
  验证：右边垂直(60.0)✓，左边垂直(42.0)✓，非矩形✓，尖朝左✓"
</expected_output>
</step_3_shape_construction>

<step_4_validation>
<objective>验证分析结果的合理性（包括聚类质量、方向一致性）</objective>

<checks>
1. 数量检查：
   - 预期热点数量：约 {estimated_hotspots} 个（输入三角形数 × 0.25）
   - 实际识别数量：? 个
   - 是否合理？如果差距过大，需要重新审视聚类

2. 覆盖率检查：
   - 总共{triangle_count}个输入三角形
   - 被包含在热点中的三角形数：? 个
   - 覆盖率应该 > 70%

3. 几何有效性：
   - 所有预测形状是否符合几何约束？
   - 三角形有垂直边？梯形有两条垂直边？
   - 是否避免了矩形（水平边）？

4. **方向一致性检查**（重要！）：
   【当前方向：{direction}】
   - 上り方向：所有热点的 back_kp > front_kp（尖朝右 ➡️）
   - 下り方向：所有热点的 back_kp < front_kp（尖朝左 ⬅️）
   - 检查每个热点的vertices是否符合方向规则

5. 优先级检查：
   - 三年数据都有的区域是否都识别了？
   - 是否有孤立的单年事件被错误识别为热点？

<expected_output>
在thinking.step4_validation中输出：
"验证结果：

1. 数量检查：识别5个热点（预期{estimated_hotspots}个）✓

2. 覆盖率检查：覆盖{triangle_count}个三角形中的28个（覆盖率80%）✓

3. 几何有效性：所有形状符合约束 ✓

4. 方向一致性：{direction}方向，所有热点朝向正确 ✓
   - Hotspot 1: back_kp=48.0 > front_kp=30.0 (上り) ✓
   - Hotspot 2: back_kp=42.0 < front_kp=60.0 (下り) ✓
   ...

5. 优先级检查：三年覆盖的3个区域全部识别 ✓

最终结论：分析结果合理且符合所有约束条件"
</expected_output>
</checks>
</step_4_validation>

</thinking_instructions>

<output_format>
请严格按照以下JSON格式输出（不要添加markdown标记）：

{{
  "thinking": {{
    "step1_clustering": "详细的聚类分析过程...",
    "step2_boundaries": {{
      "cluster_1": {{"front_kp_boundary": [30.0, 33.0], "back_kp_boundary": [42.0, 48.0], ...}},
      ...
    }},
    "step3_construction": "详细的顶点计算过程...",
    "step4_validation": "验证结果..."
  }},
  "direction": "{direction}",
  "hotspots": [
    {{
      "hotspot_id": 1,
      "kp_range": [30.0, 48.0],
      "time_range": [590, 750],
      "included_triangle_ids": [0, 3, 5, 8],
      "frequency": 4,
      "years_coverage": [2014, 2019, 2024],
      "severity": "high",
      "description": "早高峰KP30-48km区间拥堵",
      "reasoning": "这4个三角形空间距离<7km，时间距离<3小时，三年数据全覆盖",
      "prediction_shape": {{
        "shape_type": "trapezoid",
        "vertices": [[30.0, 600], [48.0, 590], [48.0, 750], [30.0, 745]]
      }}
    }}
  ],
  "summary": {{
    "total_hotspots": 1,
    "most_severe_hotspot_id": 1,
    "analysis_confidence": 0.90,
    "reasoning_summary": "基于三年数据识别出X个高概率拥堵区域"
  }}
}}
</output_format>

<important_guidelines>
1. 预期热点数量：约 {estimated_hotspots} 个（可以是{max(2, estimated_hotspots-2)}-{estimated_hotspots+3}个）

2. 优先级规则：
   - 三年数据都有 → 必须预测（severity: high）
   - 两年数据有 → 应该预测（severity: medium）
   - 只有一年且孤立 → 可以忽略

3. 覆盖率目标：≥ 70%的输入三角形应该被包含在某个热点中

4. 自我检查清单：
   ✓ 方向一致性是否正确？
   ✓ 几何形状是否符合约束？
   ✓ 三年覆盖的区域是否都识别了？
</important_guidelines>"""
        return prompt
    @staticmethod
    def get_example_output() -> Dict[str, Any]:
        """获取输出格式示例"""
        return {
            "direction": "上",
            "hotspots": [
                {
                    "hotspot_id": 1,
                    "kp_range": [23.5, 31.2],
                    "time_range": [960, 1200],
                    "included_triangle_ids": [1, 3, 5, 8, 12],
                    "frequency": 5,
                    "severity": "high",
                    "description": "下午4点到晚上8点的晚高峰拥堵"
                }
            ],
            "summary": {
                "total_hotspots": 1,
                "most_severe_hotspot_id": 1,
                "analysis_confidence": 0.85
            }
        }


# 便捷函数
def build_analysis_prompt(
    direction: str,
    csv_files: List[str],
    triangles: List[Dict[str, Any]],
    max_triangles: int = None  # None = 传输所有三角形
) -> str:
    """
    构建分析prompt的便捷函数（使用混合格式：坐标+统计信息）

    Args:
        direction: 方向
        csv_files: CSV文件列表
        triangles: 三角形数据
        max_triangles: 最多传输的三角形数量，None表示传输所有

    Returns:
        完整的prompt字符串
    """
    if not triangles:
        return ""

    # 第一步：规范化三角形数据（去除重复顶点，确保三角形3个点，四边形4个点）
    print(f"\n🔧 Normalizing {len(triangles)} shapes before LLM analysis...")
    triangles = DataPreparer.normalize_triangle_shapes(triangles)

    if not triangles:
        print("⚠️ Warning: No valid triangles after normalization")
        return ""

    # 提取统计信息
    kp_values = [t.get('kp_start', 0) for t in triangles] + [t.get('kp_end', 0) for t in triangles]
    time_values = [t.get('time_start', 0) for t in triangles] + [t.get('time_end', 0) for t in triangles]

    kp_range = [min(kp_values), max(kp_values)]
    time_range = [min(time_values), max(time_values)]

    # 三角形データを準備（ハイブリッド形式：vertices座標 + 統計情報）
    triangles_data = []
    for i, t in enumerate(triangles):
        # 提取顶点坐标
        vertices = t.get('vertices', [])

        # verticesフィールドがない場合、境界から構築を試みる（後方互換性）
        if not vertices:
            # 単純三角形頂点を構築（kp_start, kp_end, time等に基づく）
            kp_start = t.get('kp_start', 0)
            kp_end = t.get('kp_end', 0)
            time_start = t.get('time_start', 0)
            time_end = t.get('time_end', 0)
            time_peak = t.get('time_peak', (time_start + time_end) / 2)

            vertices = [
                (kp_start, time_start),
                (kp_end, time_peak),
                (kp_start, time_end)
            ]

        # 格式化顶点坐标（添加可读的时间字符串）
        formatted_vertices = []
        for v in vertices:
            kp, time_min = v
            time_hour, time_min_part = divmod(int(time_min), 60)
            formatted_vertices.append({
                "kp": round(kp, 2),
                "time_min": int(time_min),
                "time_str": f"{time_hour:02d}:{time_min_part:02d}"
            })

        # 构建混合格式的三角形数据
        triangle_data = {
            "id": i,
            "shape_type": t.get('shape_type', 'triangle'),  # 形状类型
            "vertices": formatted_vertices,  # 完整坐标
            "summary": {  # 统计摘要（便于LLM快速理解）
                "kp_range": [round(t.get('kp_start', 0), 2), round(t.get('kp_end', 0), 2)],
                "kp_span": round(t.get('width', 0) if 'width' in t else t.get('kp_end', 0) - t.get('kp_start', 0), 2),
                "time_range": [int(t.get('time_start', 0)), int(t.get('time_end', 0))],
                "duration_min": int(t.get('height', 0) if 'height' in t else t.get('time_end', 0) - t.get('time_start', 0)),
                "peak_time": int(t.get('time_peak', 0)),
                "area": round(t.get('area', 0), 2)
            }
        }

        triangles_data.append(triangle_data)

    # 限制数量（如果指定）
    has_more = False
    if max_triangles and len(triangles_data) > max_triangles:
        triangles_data = triangles_data[:max_triangles]
        has_more = True

    return PromptTemplates.get_hotspot_analysis_prompt(
        direction=direction,
        csv_files=csv_files,
        triangle_count=len(triangles),
        kp_range=kp_range,
        time_range=time_range,
        triangles_data=triangles_data,
        max_triangles_display=len(triangles_data),
        has_more=has_more
    )