"""
数据准备模块
负责将原始数据转换为LLM可以理解的格式
"""

from typing import Dict, List, Any, Optional
import pandas as pd


class DataPreparer:
    """数据准备器基类"""

    @staticmethod
    def clean_vertices(vertices: List[List[float]]) -> List[List[float]]:
        """
        清理顶点数据，去除重复的连续顶点

        Args:
            vertices: 顶点列表，每个顶点是 [kp, time]

        Returns:
            清理后的顶点列表
        """
        if not vertices or len(vertices) < 2:
            return vertices

        cleaned = []
        for vertex in vertices:
            # 检查是否与上一个顶点重复（考虑浮点数精度）
            if not cleaned or \
               abs(vertex[0] - cleaned[-1][0]) > 0.01 or \
               abs(vertex[1] - cleaned[-1][1]) > 0.01:
                cleaned.append(vertex)

        # 检查首尾是否重复（闭合多边形的情况）
        if len(cleaned) > 2:
            if abs(cleaned[0][0] - cleaned[-1][0]) < 0.01 and \
               abs(cleaned[0][1] - cleaned[-1][1]) < 0.01:
                cleaned = cleaned[:-1]

        return cleaned

    @staticmethod
    def normalize_triangle_shapes(triangles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        规范化三角形/四边形数据：
        - 去除重复顶点
        - 根据实际顶点数量更新 shape_type
        - 三角形必须有3个不同的点
        - 四边形必须有4个不同的点

        Args:
            triangles: 原始三角形数据列表

        Returns:
            规范化后的三角形数据列表
        """
        if not triangles:
            return []

        normalized = []
        stats = {
            'original_triangles': 0,
            'original_quads': 0,
            'cleaned_triangles': 0,
            'cleaned_quads': 0,
            'degraded_to_triangle': 0,
            'invalid_shapes': 0
        }

        for triangle in triangles:
            # 复制原始数据
            normalized_triangle = triangle.copy()

            # 元の shape_type を取得
            original_shape = triangle.get('shape_type', 'triangle')
            if original_shape == 'triangle':
                stats['original_triangles'] += 1
            else:
                stats['original_quads'] += 1

            # 获取顶点数据
            vertices = triangle.get('vertices', [])
            if not vertices:
                # verticesフィールドがない場合、クリーン化をスキップ
                normalized.append(normalized_triangle)
                continue

            # 清理重复顶点
            cleaned_vertices = DataPreparer.clean_vertices(vertices)
            vertex_count = len(cleaned_vertices)

            # 根据清理后的顶点数量判断形状类型
            if vertex_count < 3:
                # 无效形状（少于3个顶点）
                print(f"⚠️ Warning: Triangle {triangle.get('id', '?')} has only {vertex_count} vertices after cleaning, skipping")
                stats['invalid_shapes'] += 1
                continue
            elif vertex_count == 3:
                # 三角形
                normalized_triangle['vertices'] = cleaned_vertices
                normalized_triangle['shape_type'] = 'triangle'
                stats['cleaned_triangles'] += 1

                if original_shape == 'quadrilateral':
                    stats['degraded_to_triangle'] += 1

            elif vertex_count == 4:
                # 四边形
                normalized_triangle['vertices'] = cleaned_vertices
                normalized_triangle['shape_type'] = 'quadrilateral'
                stats['cleaned_quads'] += 1
            else:
                # 顶点数>4，保持原样但给出警告
                print(f"⚠️ Warning: Triangle {triangle.get('id', '?')} has {vertex_count} vertices (>4)")
                normalized_triangle['vertices'] = cleaned_vertices
                normalized.append(normalized_triangle)
                continue

            normalized.append(normalized_triangle)

        # 打印清理统计
        print(f"\n📊 Shape normalization statistics:")
        print(f"  Original: {stats['original_triangles']} triangles, {stats['original_quads']} quadrilaterals")
        print(f"  Cleaned: {stats['cleaned_triangles']} triangles, {stats['cleaned_quads']} quadrilaterals")
        if stats['degraded_to_triangle'] > 0:
            print(f"  ⚠️ Degraded: {stats['degraded_to_triangle']} quadrilaterals → triangles (duplicate vertices removed)")
        if stats['invalid_shapes'] > 0:
            print(f"  ❌ Invalid: {stats['invalid_shapes']} shapes removed (<3 vertices)")

        return normalized

    @staticmethod
    def prepare_triangle_data(triangles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备三角形数据，标准化格式并添加ID

        Args:
            triangles: 原始三角形数据列表

        Returns:
            标准化后的三角形数据
        """
        if not triangles:
            return []

        prepared = []
        for i, t in enumerate(triangles):
            prepared.append({
                "id": i,
                "kp_start": round(float(t.get('kp_start', 0)), 2),
                "kp_end": round(float(t.get('kp_end', 0)), 2),
                "peak_kp": round(float(t.get('peak_kp', 0)), 2),
                "time_start": int(t.get('time_start', 0)),
                "time_end": int(t.get('time_end', 0)),
                "peak_time": int(t.get('peak_time', 0))
            })

        return prepared

    @staticmethod
    def calculate_data_statistics(triangles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算数据统计信息

        Args:
            triangles: 三角形数据列表

        Returns:
            统计信息字典
        """
        if not triangles:
            return {
                "total_count": 0,
                "kp_range": [0, 0],
                "time_range": [0, 0],
                "kp_span": 0,
                "time_span": 0
            }

        # 全てのKPと時間値を抽出
        kp_values = []
        time_values = []

        for t in triangles:
            kp_values.extend([
                t.get('kp_start', 0),
                t.get('kp_end', 0),
                t.get('peak_kp', 0)
            ])
            time_values.extend([
                t.get('time_start', 0),
                t.get('time_end', 0),
                t.get('peak_time', 0)
            ])

        # 过滤无效值
        kp_values = [v for v in kp_values if v is not None and v > 0]
        time_values = [v for v in time_values if v is not None and v >= 0]

        if not kp_values or not time_values:
            return {
                "total_count": len(triangles),
                "kp_range": [0, 0],
                "time_range": [0, 0],
                "kp_span": 0,
                "time_span": 0
            }

        kp_min, kp_max = min(kp_values), max(kp_values)
        time_min, time_max = min(time_values), max(time_values)

        return {
            "total_count": len(triangles),
            "kp_range": [round(kp_min, 2), round(kp_max, 2)],
            "time_range": [int(time_min), int(time_max)],
            "kp_span": round(kp_max - kp_min, 2),
            "time_span": int(time_max - time_min)
        }

    @staticmethod
    def prepare_llm_input(
        triangles: List[Dict[str, Any]],
        direction: str,
        csv_files: List[str]
    ) -> Dict[str, Any]:
        """
        准备完整的LLM输入数据

        Args:
            triangles: 三角形数据
            direction: 方向
            csv_files: CSV文件列表

        Returns:
            完整的LLM输入数据字典
        """
        prepared_triangles = DataPreparer.prepare_triangle_data(triangles)
        statistics = DataPreparer.calculate_data_statistics(triangles)

        return {
            "direction": direction,
            "csv_files": csv_files,
            "statistics": statistics,
            "triangles": prepared_triangles
        }


class RawDataLoader:
    """原始数据加载器"""

    @staticmethod
    def load_csv_summary(csv_path: str) -> Optional[Dict[str, Any]]:
        """
        加载CSV文件并生成摘要信息

        Args:
            csv_path: CSV文件路径

        Returns:
            CSV数据摘要，如果加载失败返回None
        """
        try:
            df = pd.read_csv(csv_path)

            summary = {
                "file_name": csv_path,
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": None,
                "kp_range": None
            }

            # 尝试提取日期范围
            date_columns = [col for col in df.columns if 'date' in col.lower() or '日付' in col]
            if date_columns:
                try:
                    dates = pd.to_datetime(df[date_columns[0]])
                    summary["date_range"] = [
                        dates.min().strftime('%Y-%m-%d'),
                        dates.max().strftime('%Y-%m-%d')
                    ]
                except:
                    pass

            # KP範囲の抽出を試みる
            kp_columns = [col for col in df.columns if 'kp' in col.lower()]
            if kp_columns:
                try:
                    kp_values = df[kp_columns[0]].dropna()
                    summary["kp_range"] = [
                        float(kp_values.min()),
                        float(kp_values.max())
                    ]
                except:
                    pass

            return summary

        except Exception as e:
            print(f"⚠️ Failed to load CSV {csv_path}: {str(e)}")
            return None

    @staticmethod
    def load_multiple_csv_summaries(csv_paths: List[str]) -> List[Dict[str, Any]]:
        """
        加载多个CSV文件的摘要信息

        Args:
            csv_paths: CSV文件路径列表

        Returns:
            CSV摘要信息列表
        """
        summaries = []
        for path in csv_paths:
            summary = RawDataLoader.load_csv_summary(path)
            if summary:
                summaries.append(summary)
        return summaries


class OutputFormatter:
    """输出格式化器"""

    @staticmethod
    def format_hotspot_for_display(hotspot: Dict[str, Any]) -> str:
        """
        格式化单个热点信息用于显示

        Args:
            hotspot: 热点数据字典

        Returns:
            格式化的字符串
        """
        kp_start, kp_end = hotspot['kp_range']
        time_start, time_end = hotspot['time_range']

        # 转换时间为小时:分钟格式
        start_hour, start_min = divmod(time_start, 60)
        end_hour, end_min = divmod(time_end, 60)

        return (
            f"Hotspot {hotspot['hotspot_id']}: "
            f"KP {kp_start:.1f}-{kp_end:.1f}km, "
            f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}, "
            f"{hotspot['frequency']} events, "
            f"severity: {hotspot['severity']}"
        )

    @staticmethod
    def format_analysis_summary(analysis_result: Dict[str, Any]) -> str:
        """
        格式化分析结果摘要

        Args:
            analysis_result: LLM分析结果

        Returns:
            格式化的摘要字符串
        """
        direction = analysis_result.get('direction', 'Unknown')
        hotspots = analysis_result.get('hotspots', [])
        summary = analysis_result.get('summary', {})

        lines = [
            f"=== {direction}方向分析结果 ===",
            f"识别到 {summary.get('total_hotspots', 0)} 个热点区域",
            f"分析置信度: {summary.get('analysis_confidence', 0):.2f}"
        ]

        if hotspots:
            lines.append("\n热点详情:")
            for hotspot in hotspots:
                lines.append(f"  {OutputFormatter.format_hotspot_for_display(hotspot)}")

        return "\n".join(lines)


# 便捷函数
def prepare_direction_data(
    direction_data: Dict[str, Dict[str, Any]],
    file_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    为所有方向准备数据

    Args:
        direction_data: 方向数据字典
        file_paths: 文件路径列表

    Returns:
        准备好的数据字典
    """
    import os

    prepared = {}
    csv_files = [os.path.basename(fp) for fp in file_paths]

    for direction, data in direction_data.items():
        triangles = data.get('triangles', [])
        prepared[direction] = DataPreparer.prepare_llm_input(
            triangles=triangles,
            direction=direction,
            csv_files=csv_files
        )

    return prepared