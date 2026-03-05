"""
可視化比較モジュール
元の三角形とLLM識別ホットスポット領域を比較
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
import numpy as np
from typing import List, Dict, Any, Optional
import os


def minutes_to_time_formatter(x, pos):
    """
    Convert minutes from midnight to HH:MM format for axis labels
    Handles times beyond 24:00 (next day)
    """
    hours = int(x // 60)
    mins = int(x % 60)
    return f"{hours:02d}:{mins:02d}"


def extract_year_from_source(source_file: str) -> int:
    """
    Extract year from source_file field
    Example: '関越道_上_2014_05-05' -> 2014
    """
    if not source_file:
        return 0
    import re
    # 4桁年のパターンに一致
    match = re.search(r'_(\d{4})_', source_file)
    if match:
        return int(match.group(1))
    return 0


# 年カラー マッピング（3年の3色）
YEAR_COLORS = {
    2014: {'facecolor': '#FF6B6B', 'edgecolor': '#CC0000', 'label': '2014'},  # Red
    2019: {'facecolor': '#4ECDC4', 'edgecolor': '#008B8B', 'label': '2019'},  # Teal
    2024: {'facecolor': '#45B7D1', 'edgecolor': '#0066CC', 'label': '2024'},  # Blue
}
DEFAULT_COLOR = {'facecolor': 'lightgray', 'edgecolor': 'gray', 'label': 'Unknown'}


def convert_hotspot_to_hull(hotspot: Dict[str, Any], direction: str = "上") -> Dict[str, Any]:
    """
    Convert LLM hotspot to hull format (supports triangles and trapezoids)

    Args:
        hotspot: LLM-identified hotspot data
        direction: Direction

    Returns:
        Hull format data dictionary
    """
    # prediction_shapeを優先（新形式）
    prediction_shape = hotspot.get('prediction_shape', {})

    if prediction_shape and 'vertices' in prediction_shape:
        # 新形式：LLM予測形状を直接使用
        vertices = prediction_shape['vertices']
        shape_type = prediction_shape.get('shape_type', 'triangle')
    else:
        # レガシー形式互換性：hull_triangleから構築
        hull_triangle = hotspot.get('hull_triangle', {})

        if not hull_triangle:
            # hull_triangleもない場合、kp_rangeとtime_rangeから推定
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

        # 三角形頂点を構築
        vertices = [
            [hull_triangle['kp_start'], hull_triangle['time_start']],
            [hull_triangle['kp_end'], hull_triangle['time_end']],
            [hull_triangle['peak_kp'], hull_triangle['peak_time']]
        ]
        shape_type = 'triangle'

    # 境界ボックスと面積を計算
    kp_values = [v[0] for v in vertices]
    time_values = [v[1] for v in vertices]

    kp_min, kp_max = min(kp_values), max(kp_values)
    time_min, time_max = min(time_values), max(time_values)

    # 鞋带公式を使用してポリゴン面積を計算
    def polygon_area(vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    area = polygon_area(vertices)

    # ハull辞書を構築
    hull = {
        'cluster_id': hotspot['hotspot_id'],
        'cluster_size': hotspot['frequency'],
        'kp_range': hotspot['kp_range'],
        'time_range': hotspot['time_range'],
        'direction': direction,

        # コア：頂点座標
        'vertices': vertices,
        'shape_type': shape_type,

        # 境界情報
        'kp_start': kp_min,
        'kp_end': kp_max,
        'time_start': time_min,
        'time_end': time_max,

        # サイズと面積
        'width': kp_max - kp_min,
        'height': time_max - time_min,
        'area': area,

        # メタデータ
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
    road_name: str = "Kan-Etsu Expressway"
) -> str:
    """
    Generate comparison visualization: Original triangles vs LLM-identified hotspot regions

    Args:
        triangles: Original triangle data list
        llm_hulls: LLM-identified hotspot regions (converted to hull format)
        direction: Direction (up/down)
        output_path: Output file path
        figsize: Figure size
        road_name: Road name

    Returns:
        Saved file path
    """
    print(f"   Generating comparison visualization for {direction} direction...")

    # 図を作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # フォントを設定
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False

    # 方向名マッピング
    direction_eng = "Up" if direction == "上" else "Down"

    # ===== 左プロット：元の三角形 =====
    ax1.set_xlabel('KP (km)', fontsize=12)
    ax1.set_ylabel('Time', fontsize=12)
    ax1.set_title(f'{road_name} {direction_eng} - Historical Congestion Events', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(minutes_to_time_formatter))

    # 年ベースのカラーで元の三角形を描画
    year_legend_added = set()  # Track which years have been added to legend

    if triangles:
        for i, triangle in enumerate(triangles):
            # verticesフィールドを使用、なければ他のフィールドから構築
            if 'vertices' in triangle and triangle['vertices']:
                vertices = triangle['vertices']
            else:
                # peak_kpフィールドがない場合、ピーク位置を計算
                peak_kp = triangle.get('peak_kp', (triangle['kp_start'] + triangle['kp_end']) / 2)
                peak_time = triangle.get('peak_time', triangle.get('time_peak', (triangle['time_start'] + triangle['time_end']) / 2))

                vertices = [
                    [triangle['kp_start'], triangle['time_start']],
                    [triangle['kp_end'], triangle['time_end']],
                    [peak_kp, peak_time]
                ]

            # source_fileから年を取得し対応するカラー
            source_file = triangle.get('source_file', '')
            year = extract_year_from_source(source_file)
            color_info = YEAR_COLORS.get(year, DEFAULT_COLOR)

            # 各年の最初の出現にのみラベルを追加（凡例用）
            label = color_info['label'] if year not in year_legend_added else None
            if year:
                year_legend_added.add(year)

            poly = Polygon(
                vertices,
                alpha=0.5,
                facecolor=color_info['facecolor'],
                edgecolor=color_info['edgecolor'],
                linewidth=1.0,
                label=label
            )
            ax1.add_patch(poly)

        # 軸範囲を設定
        all_kp = [t['kp_start'] for t in triangles] + [t['kp_end'] for t in triangles]
        all_time = [t['time_start'] for t in triangles] + [t['time_end'] for t in triangles]

        if all_kp and all_time:
            kp_margin = (max(all_kp) - min(all_kp)) * 0.1
            time_margin = (max(all_time) - min(all_time)) * 0.1

            ax1.set_xlim(min(all_kp) - kp_margin, max(all_kp) + kp_margin)
            # Y軸を反転：時間は上（早期）から下（後期）へ流れる
            ax1.set_ylim(max(all_time) + time_margin, min(all_time) - time_margin)

    # 年ごとにイベントをカウント
    year_counts = {}
    for t in triangles:
        year = extract_year_from_source(t.get('source_file', ''))
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1

    # 統計テキストを構築
    stats_lines = [f'Total Events: {len(triangles)}']
    for year in sorted(year_counts.keys()):
        stats_lines.append(f'{year}: {year_counts[year]}')
    stats_text = '\n'.join(stats_lines)

    # 統計を追加
    ax1.text(
        0.02, 0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )

    # 年の凡例を追加
    if year_legend_added:
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)

    # ===== 右プロット：LLM識別ホットスポット =====
    ax2.set_xlabel('KP (km)', fontsize=12)
    ax2.set_ylabel('Time', fontsize=12)
    ax2.set_title(f'{road_name} {direction_eng} - LLM Predicted Hotspots', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(minutes_to_time_formatter))

    # 元の三角形を描画（薄い背景と年のカラー）
    if triangles:
        for triangle in triangles:
            if 'vertices' in triangle and triangle['vertices']:
                vertices = triangle['vertices']
            else:
                peak_kp = triangle.get('peak_kp', (triangle['kp_start'] + triangle['kp_end']) / 2)
                peak_time = triangle.get('peak_time', triangle.get('time_peak', (triangle['time_start'] + triangle['time_end']) / 2))

                vertices = [
                    [triangle['kp_start'], triangle['time_start']],
                    [triangle['kp_end'], triangle['time_end']],
                    [peak_kp, peak_time]
                ]

            # 年カラーを使用、ただし背景用にalphaを低く
            source_file = triangle.get('source_file', '')
            year = extract_year_from_source(source_file)
            color_info = YEAR_COLORS.get(year, DEFAULT_COLOR)

            poly = Polygon(
                vertices,
                alpha=0.25,  # More transparent for background
                facecolor=color_info['facecolor'],
                edgecolor=color_info['edgecolor'],
                linewidth=0.5
            )
            ax2.add_patch(poly)

    # LLM識別ホットスポット領域を描画（全てのホットスポットに統一カラー）
    HOTSPOT_COLOR = '#E57373'  # Unified light red color for all hotspots

    if llm_hulls:
        for i, hull in enumerate(llm_hulls):
            vertices = hull['vertices']

            # 統一カラーで塗りつぶしを描画
            poly = Polygon(
                vertices,
                alpha=0.5,
                facecolor=HOTSPOT_COLOR,
                edgecolor='darkred',
                linewidth=2.5,
                label=f'Hotspot {hull["cluster_id"]}'
            )
            ax2.add_patch(poly)

            # 注釈を追加
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

    # 同じ軸範囲を設定
    if triangles:
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())

    # 統計を追加
    ax2.text(
        0.02, 0.98,
        f'Total Hotspots: {len(llm_hulls)}\nConfidence: {llm_hulls[0].get("confidence", 0.85):.2f}' if llm_hulls else 'No hotspots detected',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )

    # 凡例を追加
    if llm_hulls:
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # レイアウトを調整
    plt.tight_layout()

    # 図を保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Comparison visualization saved: {output_path}")
    return output_path


def visualize_all_directions_comparison(
    direction_data: Dict[str, Dict[str, Any]],
    llm_analysis: Dict[str, Dict[str, Any]],
    output_dir: str = "output",
    road_name: str = "Kan-Etsu Expressway",
    timestamp: str = None
) -> Dict[str, str]:
    """
    Generate comparison visualizations for all directions

    Args:
        direction_data: Direction data dictionary
        llm_analysis: LLM analysis results
        output_dir: Output directory
        road_name: Road name
        timestamp: Timestamp

    Returns:
        Image paths dictionary for each direction
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_paths = {}

    for direction, data in direction_data.items():
        triangles = data.get('triangles', [])

        if direction not in llm_analysis:
            print(f"   No LLM analysis for {direction} direction")
            continue

        # LLMホットスポットを取得しハル形式に変換
        hotspots = llm_analysis[direction].get('hotspots', [])
        llm_hulls = [convert_hotspot_to_hull(h, direction) for h in hotspots]

        # ファイル名を生成
        direction_name = "up" if direction == "上" else "down"
        filename = f"comparison_{road_name}_{direction_name}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        # 比較可視化を生成
        saved_path = visualize_comparison(
            triangles=triangles,
            llm_hulls=llm_hulls,
            direction=direction,
            output_path=output_path,
            road_name=road_name
        )

        comparison_paths[direction] = saved_path

    return comparison_paths


# 便利な関数
def create_comparison_visualization(
    state_or_result: Dict[str, Any],
    output_dir: str = "output"
) -> Dict[str, str]:
    """
    Create comparison visualization from workflow state

    Args:
        state_or_result: Workflow state or result
        output_dir: Output directory

    Returns:
        Generated image paths dictionary
    """
    direction_data = state_or_result.get('direction_data', {})
    llm_analysis = state_or_result.get('llm_analysis', {})
    route = state_or_result.get('route', 'Kan-Etsu Expressway')

    if not direction_data or not llm_analysis:
        print("   Missing direction_data or llm_analysis")
        return {}

    return visualize_all_directions_comparison(
        direction_data=direction_data,
        llm_analysis=llm_analysis,
        output_dir=output_dir,
        road_name=route
    )
