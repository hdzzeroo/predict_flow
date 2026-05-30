#!/usr/bin/env python3
"""
正确的评估可视化 - 使用2025年Ground Truth
从evaluation结果重新加载2025年的真实数据进行可视化
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Dict, Any
import os
from datetime import datetime
from functions import process_direction_aware_traffic_data


def load_workflow_state(state_path: str) -> Dict:
    """加载workflow状态"""
    with open(state_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_2025_ground_truth(route: str, direction: str, year: int, month: int, day: int) -> List[Dict]:
    """
    加载2025年的Ground Truth数据

    Args:
        route: 道路名称
        direction: 方向
        year: 年份（2025）
        month: 月份
        day: 日期

    Returns:
        Ground Truth三角形列表
    """
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    filename = f"{route}_{direction}_{year}_{month:02d}-{day:02d}.csv"
    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(file_path):
        print(f"   ⚠️  2025年GT文件不存在: {filename}")
        return []

    print(f"   📂 加载2025年GT: {filename}")

    try:
        # 使用现有函数处理
        triangles_data, _ = process_direction_aware_traffic_data(
            file_path=file_path,
            direction=direction,
            output_dir="output/temp"
        )

        # 转换格式
        triangles = []
        for t in triangles_data:
            if 'vertices' in t and len(t.get('vertices', [])) >= 3:
                triangles.append({
                    'vertices': t['vertices'],
                    'kp_start': t.get('kp_start'),
                    'kp_end': t.get('kp_end'),
                    'time_start': t.get('time_start'),
                    'time_end': t.get('time_end'),
                    'time_peak': t.get('time_peak'),
                    'area': t.get('area')
                })

        print(f"   ✓ 加载了 {len(triangles)} 个2025年真实拥堵事件")
        return triangles

    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_correct_comparison(
    direction: str,
    gt_triangles_2025: List[Dict],
    pred_hotspots: List[Dict],
    eval_metrics: Dict,
    output_path: str,
    road_name: str = "関越道"
):
    """
    创建正确的评估对比可视化
    使用2025年的Ground Truth vs LLM预测
    """
    print(f"\n🎨 Creating correct comparison for {direction} direction...")
    print(f"   2025 Ground Truth: {len(gt_triangles_2025)} events")
    print(f"   LLM Predictions: {len(pred_hotspots)} hotspots")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形 - 3个子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax1, ax2, ax3 = axes

    # 转换为Shapely多边形
    gt_polygons = []
    pred_polygons = []

    # 处理2025年Ground Truth
    for t in gt_triangles_2025:
        if 'vertices' in t and t['vertices'] and len(t['vertices']) >= 3:
            try:
                poly = Polygon(t['vertices'])
                if poly.is_valid and poly.area > 0:
                    gt_polygons.append(poly)
            except:
                continue

    # 处理预测热点
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

    print(f"   Valid 2025 GT polygons: {len(gt_polygons)}")
    print(f"   Valid Pred polygons: {len(pred_polygons)}")

    # 计算TP/FP/FN区域
    tp_region = None
    fp_region = None
    fn_region = None
    tp_area = fp_area = fn_area = 0
    total_area = 0

    if gt_polygons and pred_polygons:
        try:
            gt_union = unary_union(gt_polygons)
            pred_union = unary_union(pred_polygons)

            tp_region = pred_union.intersection(gt_union)  # 预测正确
            fp_region = pred_union.difference(gt_union)     # 误报
            fn_region = gt_union.difference(pred_union)     # 漏检

            tp_area = tp_region.area if tp_region else 0
            fp_area = fp_region.area if fp_region else 0
            fn_area = fn_region.area if fn_region else 0
            total_area = gt_union.area + pred_union.area - tp_area

            print(f"   TP area: {tp_area:.1f} km·min (预测正确)")
            print(f"   FP area: {fp_area:.1f} km·min (误报)")
            print(f"   FN area: {fn_area:.1f} km·min (漏检)")
        except Exception as e:
            print(f"   ⚠️  计算TP/FP/FN时出错: {e}")

    # === 子图1: 2025 Ground Truth ===
    ax1.set_title(f'{road_name} {direction} - 2025年真实拥堵\n(Ground Truth: {len(gt_triangles_2025)}个事件)',
                  fontsize=13, fontweight='bold', pad=10, color='darkblue')
    ax1.set_xlabel('KP (km)', fontsize=11)
    ax1.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制2025年GT三角形
    for poly in gt_polygons:
        x, y = poly.exterior.xy
        ax1.fill(x, y, alpha=0.5, facecolor='lightblue', edgecolor='blue', linewidth=1.5)

    # 统计信息
    if gt_polygons:
        gt_union = unary_union(gt_polygons)
        stats_text = f'2025年实际:\n{len(gt_triangles_2025)} 事件\n{gt_union.area:.0f} km·min'
    else:
        stats_text = 'No 2025 data'

    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))

    # === 子图2: LLM Predictions (基于历史) ===
    ax2.set_title(f'{road_name} {direction} - LLM预测热点\n(基于历史数据: {len(pred_hotspots)}个)',
                  fontsize=13, fontweight='bold', pad=10, color='darkred')
    ax2.set_xlabel('KP (km)', fontsize=11)
    ax2.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制预测
    for poly in pred_polygons:
        x, y = poly.exterior.xy
        ax2.fill(x, y, alpha=0.5, facecolor='lightcoral', edgecolor='red', linewidth=2)

    # 统计信息
    if pred_polygons:
        pred_union = unary_union(pred_polygons)
        stats_text = f'LLM预测:\n{len(pred_hotspots)} 热点\n{pred_union.area:.0f} km·min'
    else:
        stats_text = 'No predictions'

    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='red', linewidth=2))

    # === 子图3: TP/FP/FN 分解 ===
    ax3.set_title(f'{road_name} {direction} - 评估分解\n(预测2025 vs 实际2025)',
                  fontsize=13, fontweight='bold', pad=10, color='darkgreen')
    ax3.set_xlabel('KP (km)', fontsize=11)
    ax3.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制TP (绿色)
    if tp_region and not tp_region.is_empty:
        if tp_region.geom_type == 'Polygon':
            x, y = tp_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        elif tp_region.geom_type == 'MultiPolygon':
            for poly in tp_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)

    # 绘制FP (红色)
    if fp_region and not fp_region.is_empty:
        if fp_region.geom_type == 'Polygon':
            x, y = fp_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='lightcoral', edgecolor='darkred', linewidth=2)
        elif fp_region.geom_type == 'MultiPolygon':
            for poly in fp_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='lightcoral', edgecolor='darkred', linewidth=2)

    # 绘制FN (黄色)
    if fn_region and not fn_region.is_empty:
        if fn_region.geom_type == 'Polygon':
            x, y = fn_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='yellow', edgecolor='orange', linewidth=2)
        elif fn_region.geom_type == 'MultiPolygon':
            for poly in fn_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='yellow', edgecolor='orange', linewidth=2)

    # 评估指标
    if eval_metrics and direction in eval_metrics.get('results', {}):
        metrics = eval_metrics['results'][direction]
        grid_metrics = metrics.get('grid_metrics', {})

        stats_text = (
            f"TP: {tp_area:.0f} km·min\n"
            f"FP: {fp_area:.0f} km·min\n"
            f"FN: {fn_area:.0f} km·min\n"
            f"━━━━━━━━━━━━\n"
            f"IoU: {metrics.get('polygon_iou', 0):.3f}\n"
            f"F1: {grid_metrics.get('f1_score', 0):.3f}\n"
            f"Prec: {grid_metrics.get('precision', 0):.3f}\n"
            f"Recall: {grid_metrics.get('recall', 0):.3f}"
        )
    else:
        stats_text = (
            f"TP: {tp_area:.0f}\n"
            f"FP: {fp_area:.0f}\n"
            f"FN: {fn_area:.0f}\n"
            f"IoU: {tp_area/total_area:.3f}" if total_area > 0 else "No data"
        )

    ax3.text(0.02, 0.98, stats_text,
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
             family='monospace')

    # 统一坐标轴
    if gt_polygons or pred_polygons:
        all_bounds = [p.bounds for p in (gt_polygons + pred_polygons)]
        kp_min = min(b[0] for b in all_bounds)
        kp_max = max(b[2] for b in all_bounds)
        time_min = min(b[1] for b in all_bounds)
        time_max = max(b[3] for b in all_bounds)

        kp_margin = max((kp_max - kp_min) * 0.05, 5)
        time_margin = max((time_max - time_min) * 0.05, 30)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(kp_min - kp_margin, kp_max + kp_margin)
            ax.set_ylim(time_min - time_margin, time_max + time_margin)

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', alpha=0.7, label='TP: 预测正确'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', alpha=0.7, label='FP: 误报（过度预测）'),
        mpatches.Patch(facecolor='yellow', edgecolor='orange', alpha=0.7, label='FN: 漏检（预测不足）')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved: {output_path}")
    return output_path


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🔍 Correct Evaluation Visualization (2025 GT)")
    print("   Using 2025 Ground Truth for evaluation")
    print("="*70)

    # 加载workflow state
    state_path = "output/workflow_state.json"
    eval_dir = "output/evaluation"
    output_dir = "output/correct_eval"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(state_path):
        print(f"❌ 找不到: {state_path}")
        return

    print(f"\n📂 Loading: {state_path}")
    state = load_workflow_state(state_path)

    # 提取信息
    route = state.get('route', '関越道')
    target_year = state.get('target_year', 2025)
    target_month = state.get('target_month')
    target_day = state.get('target_day')
    llm_analysis = state.get('llm_analysis', {})

    print(f"   Route: {route}")
    print(f"   Target: {target_year}/{target_month}/{target_day}")

    if not llm_analysis:
        print("❌ 缺少 llm_analysis")
        return

    # 为每个方向创建可视化
    for direction in llm_analysis.keys():
        print(f"\n{'─'*70}")
        print(f"📊 Processing: {direction} 方向")
        print(f"{'─'*70}")

        # 加载2025年GT
        gt_triangles_2025 = load_2025_ground_truth(
            route, direction, target_year, target_month, target_day
        )

        if not gt_triangles_2025:
            print(f"   ⚠️  跳过：无2025年数据")
            continue

        # 获取预测
        pred_hotspots = llm_analysis[direction].get('hotspots', [])

        # 加载评估结果
        eval_files = [f for f in os.listdir(eval_dir)
                     if f.startswith(f'evaluation_{route}_{direction}_') and f.endswith('.json')]

        eval_metrics = None
        if eval_files:
            latest_eval = sorted(eval_files)[-1]
            with open(os.path.join(eval_dir, latest_eval), 'r') as f:
                eval_metrics = json.load(f)
            print(f"   ✓ 加载评估指标: {latest_eval}")

        # 生成可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"correct_eval_{route}_{direction}_2025_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            create_correct_comparison(
                direction=direction,
                gt_triangles_2025=gt_triangles_2025,
                pred_hotspots=pred_hotspots,
                eval_metrics=eval_metrics,
                output_path=output_path,
                road_name=route
            )
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ 正确的评估可视化完成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
