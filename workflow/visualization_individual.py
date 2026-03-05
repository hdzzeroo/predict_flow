"""
個別可視化モジュール
実験結果から統一スケールで5つの別々の図を作成

図:
1. 歴史的交通拥堵イベント（2014/2019/2024から）
2. LLM予測ホットスポット
3. グラウンドトゥルース（2025実データ）
4. 予測（評価ビュー用）
5. TP/FP/FN分解
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def minutes_to_time_formatter(x, pos):
    """深夜0時からの分をHH:MM形式に変換"""
    hours = int(x // 60)
    mins = int(x % 60)
    return f"{hours:02d}:{mins:02d}"


def extract_year_from_source(source_file: str) -> int:
    """source_fileフィールドから年を抽出"""
    if not source_file:
        return 0
    import re
    match = re.search(r'_(\d{4})_', source_file)
    if match:
        return int(match.group(1))
    return 0


# 年カラー マッピング
YEAR_COLORS = {
    2014: {'facecolor': '#FF6B6B', 'edgecolor': '#CC0000', 'label': '2014'},
    2019: {'facecolor': '#4ECDC4', 'edgecolor': '#008B8B', 'label': '2019'},
    2024: {'facecolor': '#45B7D1', 'edgecolor': '#0066CC', 'label': '2024'},
}
DEFAULT_COLOR = {'facecolor': 'lightgray', 'edgecolor': 'gray', 'label': 'Unknown'}


def load_experiment_result(result_path: str) -> Dict[str, Any]:
    """Load experiment result from JSON file"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_historical_triangles(
    route: str,
    direction: str,
    historical_years: List[int],
    target_month: int,
    target_day: int,
    date_matching_strategy: str = "same_weekday",
    base_dir: str = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
) -> List[Dict]:
    """
    Load historical triangle data from CSV files

    Args:
        route: Road name (e.g., "関越道")
        direction: Direction ("上" or "下")
        historical_years: List of historical years [2014, 2019, 2024]
        target_month: Target month
        target_day: Target day
        date_matching_strategy: "same_date" or "same_weekday"
        base_dir: Base directory for data files

    Returns:
        List of triangle dictionaries
    """
    from workflow.functions import process_direction_aware_traffic_data, get_matched_file_date

    all_triangles = []

    for year in historical_years:
        # 戦略に基づいて一致した日付を取得
        file_month, file_day = get_matched_file_date(
            target_year=2025,  # Target year for matching
            target_month=target_month,
            target_day=target_day,
            historical_year=year,
            strategy=date_matching_strategy
        )

        filename = f"{route}_{direction}_{year}_{file_month:02d}-{file_day:02d}.csv"
        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):
            print(f"   File not found: {filename}")
            continue

        print(f"   Loading: {filename}")

        try:
            triangles_data, _ = process_direction_aware_traffic_data(
                file_path=file_path,
                direction=direction,
                output_dir="output/temp"
            )

            for t in triangles_data:
                t['source_file'] = f"{route}_{direction}_{year}_{file_month:02d}-{file_day:02d}"
                all_triangles.append(t)

        except Exception as e:
            print(f"   Error loading {filename}: {e}")

    print(f"   Total historical triangles: {len(all_triangles)}")
    return all_triangles


def load_ground_truth(
    route: str,
    direction: str,
    year: int,
    month: int,
    day: int,
    base_dir: str = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
) -> List[Dict]:
    """Load ground truth data for a specific date"""
    from workflow.functions import process_direction_aware_traffic_data

    filename = f"{route}_{direction}_{year}_{month:02d}-{day:02d}.csv"
    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(file_path):
        print(f"   Ground truth file not found: {filename}")
        return []

    print(f"   Loading ground truth: {filename}")

    try:
        triangles_data, _ = process_direction_aware_traffic_data(
            file_path=file_path,
            direction=direction,
            output_dir="output/temp"
        )

        triangles = []
        for t in triangles_data:
            if 'vertices' in t and len(t.get('vertices', [])) >= 3:
                triangles.append(t)

        print(f"   Ground truth events: {len(triangles)}")
        return triangles

    except Exception as e:
        print(f"   Error loading ground truth: {e}")
        return []


def calculate_unified_bounds(
    historical_triangles: List[Dict],
    pred_hotspots: List[Dict],
    gt_triangles: List[Dict]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate unified axis bounds for all figures

    Returns:
        ((kp_min, kp_max), (time_min, time_max))
    """
    all_kp = []
    all_time = []

    # 歴史的三角形
    for t in historical_triangles:
        if 'vertices' in t:
            for v in t['vertices']:
                all_kp.append(v[0])
                all_time.append(v[1])
        else:
            all_kp.extend([t.get('kp_start', 0), t.get('kp_end', 0)])
            all_time.extend([t.get('time_start', 0), t.get('time_end', 0)])

    # 予測
    for h in pred_hotspots:
        if 'prediction_shape' in h and 'vertices' in h['prediction_shape']:
            for v in h['prediction_shape']['vertices']:
                all_kp.append(v[0])
                all_time.append(v[1])

    # グラウンドトゥルース
    for t in gt_triangles:
        if 'vertices' in t:
            for v in t['vertices']:
                all_kp.append(v[0])
                all_time.append(v[1])

    if not all_kp or not all_time:
        return ((0, 100), (0, 1440))

    kp_min, kp_max = min(all_kp), max(all_kp)
    time_min, time_max = min(all_time), max(all_time)

    # マージンを追加
    kp_margin = max((kp_max - kp_min) * 0.05, 5)
    time_margin = max((time_max - time_min) * 0.05, 30)

    return (
        (kp_min - kp_margin, kp_max + kp_margin),
        (time_min - time_margin, time_max + time_margin)
    )


def create_single_figure(
    ax,
    title: str,
    kp_bounds: Tuple[float, float],
    time_bounds: Tuple[float, float],
    road_name: str = "Kan-Etsu Expressway",
    direction_eng: str = "Up"
):
    """Set up a single figure with consistent styling"""
    ax.set_xlabel('KP (km)', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(minutes_to_time_formatter))

    ax.set_xlim(kp_bounds)
    # Y軸を反転：時間は上（早期）から下（後期）へ流れる
    ax.set_ylim(time_bounds[1], time_bounds[0])


def draw_triangles(ax, triangles: List[Dict], use_year_colors: bool = True,
                   alpha: float = 0.5, single_color: str = None):
    """Draw triangle/polygon shapes on axis"""
    year_legend_added = set()

    for t in triangles:
        if 'vertices' in t and t['vertices']:
            vertices = t['vertices']
        else:
            peak_kp = t.get('peak_kp', (t['kp_start'] + t['kp_end']) / 2)
            peak_time = t.get('peak_time', t.get('time_peak', (t['time_start'] + t['time_end']) / 2))
            vertices = [
                [t['kp_start'], t['time_start']],
                [t['kp_end'], t['time_end']],
                [peak_kp, peak_time]
            ]

        if single_color:
            facecolor = single_color
            edgecolor = 'darkblue' if single_color == 'lightblue' else 'darkred'
            label = None
        elif use_year_colors:
            source_file = t.get('source_file', '')
            year = extract_year_from_source(source_file)
            color_info = YEAR_COLORS.get(year, DEFAULT_COLOR)
            facecolor = color_info['facecolor']
            edgecolor = color_info['edgecolor']
            label = color_info['label'] if year not in year_legend_added else None
            if year:
                year_legend_added.add(year)
        else:
            facecolor = 'lightgray'
            edgecolor = 'gray'
            label = None

        poly = MplPolygon(
            vertices,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
            label=label
        )
        ax.add_patch(poly)

    return year_legend_added


def draw_predictions(ax, hotspots: List[Dict], alpha: float = 0.5):
    """Draw LLM prediction hotspots"""
    HOTSPOT_COLOR = '#E57373'

    for i, h in enumerate(hotspots):
        if 'prediction_shape' in h and 'vertices' in h['prediction_shape']:
            vertices = h['prediction_shape']['vertices']

            poly = MplPolygon(
                vertices,
                alpha=alpha,
                facecolor=HOTSPOT_COLOR,
                edgecolor='darkred',
                linewidth=2.5,
                label=f'Hotspot {h["hotspot_id"]}' if i < 5 else None
            )
            ax.add_patch(poly)

            # 注釈を追加
            center_kp = np.mean([v[0] for v in vertices])
            center_time = np.mean([v[1] for v in vertices])

            ax.text(
                center_kp, center_time,
                f'#{h["hotspot_id"]}\n{h["frequency"]}x',
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )


def create_individual_visualizations(
    experiment_result_path: str,
    direction: str = "上",
    output_dir: str = None,
    figsize: Tuple[int, int] = (10, 8),
    target_year: int = 2025
) -> Dict[str, str]:
    """
    Create 5 individual visualization figures with unified scale

    Args:
        experiment_result_path: Path to experiment_result.json
        direction: Direction to visualize ("上" or "下")
        output_dir: Output directory (defaults to same as experiment result)
        figsize: Figure size for each individual plot
        target_year: Target prediction year for ground truth

    Returns:
        Dictionary mapping figure name to output path
    """
    print(f"\n{'='*70}")
    print(f"Creating individual visualizations for {direction} direction")
    print(f"{'='*70}")

    # フォントを設定
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False

    # 実験結果を読み込む
    result = load_experiment_result(experiment_result_path)
    config = result.get('config', {})
    predictions = result.get('predictions', {})
    evaluation = result.get('evaluation', {})

    # 設定を抽出
    route = config.get('data', {}).get('road', '関越道')
    target_date = config.get('data', {}).get('target_date', '2025-05-05')
    historical_years = config.get('data', {}).get('historical_years', [2014, 2019, 2024])
    date_matching_strategy = config.get('data', {}).get('date_matching_strategy', 'same_weekday')

    # 目標日付を解析
    parts = target_date.split('-')
    target_month = int(parts[1])
    target_day = int(parts[2])

    direction_eng = "Up" if direction == "上" else "Down"
    road_name = "Kan-Etsu Expressway" if route == "関越道" else route

    # 出力ディレクトリを設定
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(experiment_result_path), "individual_figures")
    os.makedirs(output_dir, exist_ok=True)

    # 全データを読み込む
    print("\n📂 Loading data...")

    # 歴史的三角形
    historical_triangles = load_historical_triangles(
        route=route,
        direction=direction,
        historical_years=historical_years,
        target_month=target_month,
        target_day=target_day,
        date_matching_strategy=date_matching_strategy
    )

    # 予測
    pred_hotspots = predictions.get(direction, {}).get('hotspots', [])
    print(f"   LLM predictions: {len(pred_hotspots)} hotspots")

    # グラウンドトゥルース
    gt_triangles = load_ground_truth(
        route=route,
        direction=direction,
        year=target_year,
        month=target_month,
        day=target_day
    )

    # 統一境界を計算
    kp_bounds, time_bounds = calculate_unified_bounds(
        historical_triangles, pred_hotspots, gt_triangles
    )
    print(f"\n📐 Unified bounds:")
    print(f"   KP: {kp_bounds[0]:.1f} - {kp_bounds[1]:.1f} km")
    print(f"   Time: {int(time_bounds[0])} - {int(time_bounds[1])} min")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths = {}

    # タイトル用の日付文字列をフォーマット
    date_str = f"{target_month}/{target_day}"

    # ========== 図1：歴史的交通拥堵イベント ==========
    print("\n🎨 Creating Figure 1: Historical Events...")
    fig1, ax1 = plt.subplots(figsize=figsize)
    create_single_figure(
        ax1,
        f'{road_name} {direction_eng} - Historical Congestion Events ({date_str})\n(2014, 2019, 2024)',
        kp_bounds, time_bounds, road_name, direction_eng
    )

    year_legend = draw_triangles(ax1, historical_triangles, use_year_colors=True)

    # 統計
    year_counts = {}
    for t in historical_triangles:
        year = extract_year_from_source(t.get('source_file', ''))
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1

    stats_lines = [f'Total Events: {len(historical_triangles)}']
    for year in sorted(year_counts.keys()):
        stats_lines.append(f'{year}: {year_counts[year]}')

    ax1.text(0.02, 0.98, '\n'.join(stats_lines),
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if year_legend:
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path1 = os.path.join(output_dir, f"fig1_historical_{route}_{direction_eng}_{timestamp}.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['historical'] = path1
    print(f"   Saved: {path1}")

    # ========== 図2：LLM予測 ==========
    print("\n🎨 Creating Figure 2: LLM Predictions...")
    fig2, ax2 = plt.subplots(figsize=figsize)
    create_single_figure(
        ax2,
        f'{road_name} {direction_eng} - LLM Predicted Hotspots ({date_str})\n({len(pred_hotspots)} hotspots)',
        kp_bounds, time_bounds, road_name, direction_eng
    )

    # 歴史を背景として描画
    draw_triangles(ax2, historical_triangles, use_year_colors=True, alpha=0.2)

    # 予測を描画
    draw_predictions(ax2, pred_hotspots, alpha=0.6)

    ax2.text(0.02, 0.98, f'Predictions: {len(pred_hotspots)} hotspots',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    if pred_hotspots:
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path2 = os.path.join(output_dir, f"fig2_prediction_{route}_{direction_eng}_{timestamp}.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['prediction'] = path2
    print(f"   Saved: {path2}")

    # ========== 図3：グラウンドトゥルース (2025) ==========
    print("\n🎨 Creating Figure 3: Ground Truth...")
    fig3, ax3 = plt.subplots(figsize=figsize)
    create_single_figure(
        ax3,
        f'{road_name} {direction_eng} - {target_year}/{date_str} Ground Truth\n({len(gt_triangles)} actual events)',
        kp_bounds, time_bounds, road_name, direction_eng
    )

    draw_triangles(ax3, gt_triangles, use_year_colors=False,
                   alpha=0.5, single_color='lightblue')

    ax3.text(0.02, 0.98, f'Actual {target_year}: {len(gt_triangles)} events',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue'))

    plt.tight_layout()
    path3 = os.path.join(output_dir, f"fig3_ground_truth_{route}_{direction_eng}_{target_year}_{timestamp}.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['ground_truth'] = path3
    print(f"   Saved: {path3}")

    # ========== 図4：予測（評価ビュー） ==========
    print("\n🎨 Creating Figure 4: Predictions (Eval View)...")
    fig4, ax4 = plt.subplots(figsize=figsize)
    create_single_figure(
        ax4,
        f'{road_name} {direction_eng} - LLM Predictions ({date_str})\n(For comparison with {target_year}/{date_str} actual)',
        kp_bounds, time_bounds, road_name, direction_eng
    )

    # 評価と同じスタイルで予測を描画
    for h in pred_hotspots:
        if 'prediction_shape' in h and 'vertices' in h['prediction_shape']:
            vertices = h['prediction_shape']['vertices']
            poly = MplPolygon(
                vertices,
                alpha=0.5,
                facecolor='lightcoral',
                edgecolor='red',
                linewidth=2
            )
            ax4.add_patch(poly)

    ax4.text(0.02, 0.98, f'LLM Predictions: {len(pred_hotspots)} hotspots',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='red'))

    plt.tight_layout()
    path4 = os.path.join(output_dir, f"fig4_prediction_eval_{route}_{direction_eng}_{timestamp}.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['prediction_eval'] = path4
    print(f"   Saved: {path4}")

    # ========== 図5：TP/FP/FN分解 ==========
    print("\n🎨 Creating Figure 5: TP/FP/FN Decomposition...")
    fig5, ax5 = plt.subplots(figsize=figsize)
    create_single_figure(
        ax5,
        f'{road_name} {direction_eng} - Evaluation Decomposition ({date_str})\n(Predicted vs Actual {target_year}/{date_str})',
        kp_bounds, time_bounds, road_name, direction_eng
    )

    # Shapelyポリゴンに変換
    gt_polygons = []
    pred_polygons = []

    for t in gt_triangles:
        if 'vertices' in t and t['vertices'] and len(t['vertices']) >= 3:
            try:
                poly = Polygon(t['vertices'])
                if poly.is_valid and poly.area > 0:
                    gt_polygons.append(poly)
            except:
                continue

    for h in pred_hotspots:
        if 'prediction_shape' in h and 'vertices' in h['prediction_shape']:
            vertices = h['prediction_shape']['vertices']
            if vertices and len(vertices) >= 3:
                try:
                    poly = Polygon(vertices)
                    if poly.is_valid and poly.area > 0:
                        pred_polygons.append(poly)
                except:
                    continue

    # TP/FP/FNを計算
    tp_area = fp_area = fn_area = 0

    if gt_polygons and pred_polygons:
        try:
            gt_union = unary_union(gt_polygons)
            pred_union = unary_union(pred_polygons)

            tp_region = pred_union.intersection(gt_union)
            fp_region = pred_union.difference(gt_union)
            fn_region = gt_union.difference(pred_union)

            tp_area = tp_region.area if tp_region else 0
            fp_area = fp_region.area if fp_region else 0
            fn_area = fn_region.area if fn_region else 0

            # 領域を描画
            def draw_region(ax, region, facecolor, edgecolor):
                if region and not region.is_empty:
                    if region.geom_type == 'Polygon':
                        x, y = region.exterior.xy
                        ax.fill(x, y, alpha=0.7, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
                    elif region.geom_type == 'MultiPolygon':
                        for poly in region.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, alpha=0.7, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)

            draw_region(ax5, tp_region, 'lightgreen', 'darkgreen')
            draw_region(ax5, fp_region, 'lightcoral', 'darkred')
            draw_region(ax5, fn_region, 'yellow', 'orange')

        except Exception as e:
            print(f"   Error calculating TP/FP/FN: {e}")

    # 指標を計算
    total_area = tp_area + fp_area + fn_area
    poly_iou = tp_area / (tp_area + fp_area + fn_area) if total_area > 0 else 0
    poly_precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0
    poly_recall = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0
    poly_f1 = 2 * poly_precision * poly_recall / (poly_precision + poly_recall) if (poly_precision + poly_recall) > 0 else 0

    # 評価からグリッドベースの指標を取得
    grid_metrics = {}
    if direction in evaluation:
        dir_eval = evaluation[direction]
        if direction in dir_eval:
            grid_metrics = dir_eval[direction].get('grid_metrics', {})
        elif 'average' in dir_eval:
            grid_metrics = dir_eval['average'].get('grid_metrics', {})

    # 統計テキスト
    stats_text = (
        f"TP: {tp_area:.0f} km*min\n"
        f"FP: {fp_area:.0f} km*min\n"
        f"FN: {fn_area:.0f} km*min\n"
        f"------------\n"
        f"[Polygon-based]\n"
        f"IoU: {poly_iou:.3f}\n"
        f"F1: {poly_f1:.3f}\n"
        f"Prec: {poly_precision:.3f}\n"
        f"Recall: {poly_recall:.3f}\n"
        f"------------\n"
        f"[Grid-based]\n"
        f"IoU: {grid_metrics.get('grid_iou', 0):.3f}\n"
        f"F1: {grid_metrics.get('f1_score', 0):.3f}\n"
        f"Prec: {grid_metrics.get('precision', 0):.3f}\n"
        f"Recall: {grid_metrics.get('recall', 0):.3f}"
    )

    ax5.text(0.02, 0.98, stats_text,
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             family='monospace')

    # 凡例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', alpha=0.7, label='TP: Correct'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', alpha=0.7, label='FP: False Positive'),
        mpatches.Patch(facecolor='yellow', edgecolor='orange', alpha=0.7, label='FN: Missed')
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path5 = os.path.join(output_dir, f"fig5_tp_fp_fn_{route}_{direction_eng}_{target_year}_{timestamp}.png")
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['tp_fp_fn'] = path5
    print(f"   Saved: {path5}")

    print(f"\n✅ All 5 figures created successfully!")
    print(f"📁 Output directory: {output_dir}")

    return output_paths


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Create individual visualizations from experiment results')
    parser.add_argument('result_path', type=str, help='Path to experiment_result.json')
    parser.add_argument('--direction', type=str, default='上', choices=['上', '下'],
                        help='Direction to visualize')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 8],
                        help='Figure size (width height)')
    parser.add_argument('--target-year', type=int, default=2025,
                        help='Target year for ground truth')

    args = parser.parse_args()

    create_individual_visualizations(
        experiment_result_path=args.result_path,
        direction=args.direction,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize),
        target_year=args.target_year
    )


if __name__ == "__main__":
    main()
