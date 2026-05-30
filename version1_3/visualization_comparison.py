"""
可视化对比模块
用于对比原始三角形和LLM识别的热点区域
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from typing import List, Dict, Any, Optional
import os


def convert_hotspot_to_hull(hotspot: Dict[str, Any], direction: str = "上") -> Dict[str, Any]:
    """
    将LLM的hotspot转换为hull格式（支持三角形和梯形）

    Args:
        hotspot: LLM识别的热点数据
        direction: 方向

    Returns:
        hull格式的数据字典
    """
    # 优先使用prediction_shape（新格式）
    prediction_shape = hotspot.get('prediction_shape', {})

    if prediction_shape and 'vertices' in prediction_shape:
        # 新格式：直接使用LLM预测的形状
        vertices = prediction_shape['vertices']
        shape_type = prediction_shape.get('shape_type', 'triangle')
    else:
        # 旧格式兼容：从hull_triangle构造
        hull_triangle = hotspot.get('hull_triangle', {})

        if not hull_triangle:
            # 如果hull_triangle也没有，从kp_range和time_range估算
            kp_range = hotspot['kp_range']
            time_range = hotspot['time_range']
            kp_center = (kp_range[0] + kp_range[1]) / 2
            time_center = (time_range[0] + time_range[1]) / 2

            hull_triangle = {
                'kp_start': kp_range[0],
                'kp_end': kp_range[1],
                'peak_kp': kp_center,
                'time_start': time_range[0],
                'time_end': time_range[1],
                'peak_time': time_center
            }

        # 构造三角形顶点
        vertices = [
            [hull_triangle['kp_start'], hull_triangle['time_start']],
            [hull_triangle['kp_end'], hull_triangle['time_end']],
            [hull_triangle['peak_kp'], hull_triangle['peak_time']]
        ]
        shape_type = 'triangle'

    # 计算边界框和面积
    kp_values = [v[0] for v in vertices]
    time_values = [v[1] for v in vertices]

    kp_min, kp_max = min(kp_values), max(kp_values)
    time_min, time_max = min(time_values), max(time_values)

    # 使用鞋带公式计算多边形面积
    def polygon_area(vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    area = polygon_area(vertices)

    # 构造hull字典
    hull = {
        'cluster_id': hotspot['hotspot_id'],
        'cluster_size': hotspot['frequency'],
        'kp_range': hotspot['kp_range'],
        'time_range': hotspot['time_range'],
        'direction': direction,

        # 核心：顶点坐标
        'vertices': vertices,
        'shape_type': shape_type,

        # 边界信息
        'kp_start': kp_min,
        'kp_end': kp_max,
        'time_start': time_min,
        'time_end': time_max,

        # 尺寸和面积
        'width': kp_max - kp_min,
        'height': time_max - time_min,
        'area': area,

        # 元数据
        'severity': hotspot.get('severity', 'medium'),
        'description': hotspot.get('description', ''),
        'reasoning': hotspot.get('reasoning', ''),
        'years_coverage': hotspot.get('years_coverage', []),
        'source': 'llm'
    }

    return hull


def visualize_comparison(
    triangles: List[Dict[str, Any]],
    llm_hulls: List[Dict[str, Any]],
    direction: str,
    output_path: str,
    figsize: tuple = (16, 12),
    road_name: str = "関越道"
) -> str:
    """
    生成对比可视化图：原始三角形 vs LLM识别的热点区域

    Args:
        triangles: 原始三角形数据列表
        llm_hulls: LLM识别的热点区域（已转换为hull格式）
        direction: 方向（上/下）
        output_path: 输出文件路径
        figsize: 图片大小
        road_name: 道路名称

    Returns:
        保存的文件路径
    """
    print(f"🎨 Generating comparison visualization for {direction} direction...")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ===== 左图：原始三角形 =====
    ax1.set_xlabel('KP (km)', fontsize=12)
    ax1.set_ylabel('Time (minutes from midnight)', fontsize=12)
    ax1.set_title(f'{road_name} {direction}り方向 - 原始拥堵事件', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 绘制原始三角形
    if triangles:
        for i, triangle in enumerate(triangles):
            # 使用vertices字段（如果存在），否则从其他字段构造
            if 'vertices' in triangle and triangle['vertices']:
                vertices = triangle['vertices']
            else:
                # 计算峰值位置（如果没有peak_kp字段）
                peak_kp = triangle.get('peak_kp', (triangle['kp_start'] + triangle['kp_end']) / 2)
                peak_time = triangle.get('peak_time', triangle.get('time_peak', (triangle['time_start'] + triangle['time_end']) / 2))

                vertices = [
                    [triangle['kp_start'], triangle['time_start']],
                    [triangle['kp_end'], triangle['time_end']],
                    [peak_kp, peak_time]
                ]

            poly = Polygon(
                vertices,
                alpha=0.4,
                facecolor='lightblue',
                edgecolor='blue',
                linewidth=1.0
            )
            ax1.add_patch(poly)

        # 设置坐标轴范围
        all_kp = [t['kp_start'] for t in triangles] + [t['kp_end'] for t in triangles]
        all_time = [t['time_start'] for t in triangles] + [t['time_end'] for t in triangles]

        if all_kp and all_time:
            kp_margin = (max(all_kp) - min(all_kp)) * 0.1
            time_margin = (max(all_time) - min(all_time)) * 0.1

            ax1.set_xlim(min(all_kp) - kp_margin, max(all_kp) + kp_margin)
            ax1.set_ylim(min(all_time) - time_margin, max(all_time) + time_margin)

    # 添加统计信息
    ax1.text(
        0.02, 0.98,
        f'Total Events: {len(triangles)}',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # ===== 右图：LLM识别的热点 =====
    ax2.set_xlabel('KP (km)', fontsize=12)
    ax2.set_ylabel('Time (minutes from midnight)', fontsize=12)
    ax2.set_title(f'{road_name} {direction}り方向 - LLM识别热点区域', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 先绘制原始三角形（浅色背景）
    if triangles:
        for triangle in triangles:
            # 使用vertices字段（如果存在），否则从其他字段构造
            if 'vertices' in triangle and triangle['vertices']:
                vertices = triangle['vertices']
            else:
                # 计算峰值位置（如果没有peak_kp字段）
                peak_kp = triangle.get('peak_kp', (triangle['kp_start'] + triangle['kp_end']) / 2)
                peak_time = triangle.get('peak_time', triangle.get('time_peak', (triangle['time_start'] + triangle['time_end']) / 2))

                vertices = [
                    [triangle['kp_start'], triangle['time_start']],
                    [triangle['kp_end'], triangle['time_end']],
                    [peak_kp, peak_time]
                ]

            poly = Polygon(
                vertices,
                alpha=0.2,
                facecolor='lightgray',
                edgecolor='gray',
                linewidth=0.5
            )
            ax2.add_patch(poly)

    # 绘制LLM识别的热点区域
    severity_colors = {
        'low': '#90EE90',    # 浅绿色
        'medium': '#FFD700',  # 金色
        'high': '#FF6B6B'     # 红色
    }

    if llm_hulls:
        for i, hull in enumerate(llm_hulls):
            vertices = hull['vertices']
            severity = hull.get('severity', 'medium')
            color = severity_colors.get(severity, '#FFD700')

            # 绘制填充
            poly = Polygon(
                vertices,
                alpha=0.5,
                facecolor=color,
                edgecolor='darkred',
                linewidth=2.5,
                label=f'Hotspot {hull["cluster_id"]} ({severity})'
            )
            ax2.add_patch(poly)

            # 添加标注
            center_kp = np.mean([v[0] for v in vertices])
            center_time = np.mean([v[1] for v in vertices])

            ax2.text(
                center_kp, center_time,
                f'#{hull["cluster_id"]}\n{hull["cluster_size"]}x',
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

    # 设置相同的坐标轴范围
    if triangles:
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())

    # 添加统计信息
    ax2.text(
        0.02, 0.98,
        f'Total Hotspots: {len(llm_hulls)}\nConfidence: {llm_hulls[0].get("confidence", 0.85):.2f}' if llm_hulls else 'No hotspots detected',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )

    # 添加图例
    if llm_hulls:
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison visualization saved: {output_path}")
    return output_path


def visualize_all_directions_comparison(
    direction_data: Dict[str, Dict[str, Any]],
    llm_analysis: Dict[str, Dict[str, Any]],
    output_dir: str = "output",
    road_name: str = "関越道",
    timestamp: str = None
) -> Dict[str, str]:
    """
    为所有方向生成对比可视化

    Args:
        direction_data: 方向数据字典
        llm_analysis: LLM分析结果
        output_dir: 输出目录
        road_name: 道路名称
        timestamp: 时间戳

    Returns:
        各方向的图片路径字典
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_paths = {}

    for direction, data in direction_data.items():
        triangles = data.get('triangles', [])

        if direction not in llm_analysis:
            print(f"⚠️ No LLM analysis for {direction} direction")
            continue

        # 获取LLM热点并转换为hull格式
        hotspots = llm_analysis[direction].get('hotspots', [])
        llm_hulls = [convert_hotspot_to_hull(h, direction) for h in hotspots]

        # 生成文件名
        direction_name = "上り" if direction == "上" else "下り"
        filename = f"comparison_{road_name}_{direction_name}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        # 生成对比图
        saved_path = visualize_comparison(
            triangles=triangles,
            llm_hulls=llm_hulls,
            direction=direction,
            output_path=output_path,
            road_name=road_name
        )

        comparison_paths[direction] = saved_path

    return comparison_paths


# 便捷函数
def create_comparison_visualization(
    state_or_result: Dict[str, Any],
    output_dir: str = "output"
) -> Dict[str, str]:
    """
    从工作流状态创建对比可视化

    Args:
        state_or_result: 工作流状态或结果
        output_dir: 输出目录

    Returns:
        生成的图片路径字典
    """
    direction_data = state_or_result.get('direction_data', {})
    llm_analysis = state_or_result.get('llm_analysis', {})
    route = state_or_result.get('route', '関越道')

    if not direction_data or not llm_analysis:
        print("⚠️ Missing direction_data or llm_analysis")
        return {}

    return visualize_all_directions_comparison(
        direction_data=direction_data,
        llm_analysis=llm_analysis,
        output_dir=output_dir,
        road_name=route
    )