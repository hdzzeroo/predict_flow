#!/usr/bin/env python3
"""
基于已有结果的评估可视化
从 workflow_state.json 和 evaluation/*.json 读取数据，生成详细的对比图
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime


def load_workflow_state(state_path: str) -> Dict:
    """加载workflow状态"""
    with open(state_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_evaluation_results(eval_dir: str, route: str, direction: str) -> Dict:
    """加载评估结果"""
    # 查找最新的评估文件
    eval_files = [f for f in os.listdir(eval_dir) if f.startswith(f'evaluation_{route}_{direction}_') and f.endswith('.json')]

    if not eval_files:
        return None

    # 按文件名排序，获取最新的
    latest_file = sorted(eval_files)[-1]
    file_path = os.path.join(eval_dir, latest_file)

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_enhanced_comparison(
    direction: str,
    gt_triangles: List[Dict],
    pred_hotspots: List[Dict],
    eval_metrics: Dict,
    output_path: str,
    road_name: str = "関越道"
):
    """
    创建增强的评估对比可视化

    三个子图：
    1. Ground Truth (真实拥堵)
    2. Predictions (LLM预测)
    3. TP/FP/FN 分解
    """
    print(f"\n🎨 Creating enhanced comparison for {direction} direction...")
    print(f"   Ground Truth: {len(gt_triangles)} triangles")
    print(f"   Predictions: {len(pred_hotspots)} hotspots")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形 - 3个子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax1, ax2, ax3 = axes

    # 转换为Shapely多边形
    gt_polygons = []
    pred_polygons = []

    # 处理Ground Truth三角形
    for t in gt_triangles:
        if 'vertices' in t and t['vertices'] and len(t['vertices']) >= 3:
            try:
                poly = Polygon(t['vertices'])
                if poly.is_valid and poly.area > 0:
                    gt_polygons.append(poly)
            except Exception as e:
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
                except Exception as e:
                    continue

    print(f"   Valid GT polygons: {len(gt_polygons)}")
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

            print(f"   TP area: {tp_area:.1f} km·min")
            print(f"   FP area: {fp_area:.1f} km·min")
            print(f"   FN area: {fn_area:.1f} km·min")
        except Exception as e:
            print(f"   ⚠️  计算TP/FP/FN时出错: {e}")

    # === 子图1: Ground Truth ===
    ax1.set_title(f'{road_name} {direction} - Ground Truth\n真实拥堵事件: {len(gt_triangles)}个',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('KP (km)', fontsize=11)
    ax1.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制所有GT三角形
    for i, poly in enumerate(gt_polygons):
        x, y = poly.exterior.xy
        ax1.fill(x, y, alpha=0.5, facecolor='lightblue', edgecolor='blue', linewidth=1.5)

    # 统计信息
    if gt_polygons:
        gt_union = unary_union(gt_polygons)
        stats_text = f'Events: {len(gt_triangles)}\nCoverage: {gt_union.area:.0f} km·min'
    else:
        stats_text = 'No data'

    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='blue'))

    # === 子图2: Predictions ===
    ax2.set_title(f'{road_name} {direction} - LLM Predictions\nLLM预测热点: {len(pred_hotspots)}个',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('KP (km)', fontsize=11)
    ax2.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制所有预测矩形/梯形
    for poly in pred_polygons:
        x, y = poly.exterior.xy
        ax2.fill(x, y, alpha=0.5, facecolor='lightcoral', edgecolor='red', linewidth=2)

    # 统计信息
    if pred_polygons:
        pred_union = unary_union(pred_polygons)
        stats_text = f'Hotspots: {len(pred_hotspots)}\nCoverage: {pred_union.area:.0f} km·min'
    else:
        stats_text = 'No predictions'

    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, edgecolor='red'))

    # === 子图3: TP/FP/FN 分解 ===
    ax3.set_title(f'{road_name} {direction} - Evaluation Breakdown\nTP/FP/FN 区域分解',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('KP (km)', fontsize=11)
    ax3.set_ylabel('Time (minutes from midnight)', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 绘制TP区域 (正确预测 - 绿色)
    if tp_region and not tp_region.is_empty:
        if tp_region.geom_type == 'Polygon':
            x, y = tp_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        elif tp_region.geom_type == 'MultiPolygon':
            for poly in tp_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)

    # 绘制FP区域 (误报 - 红色)
    if fp_region and not fp_region.is_empty:
        if fp_region.geom_type == 'Polygon':
            x, y = fp_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='lightcoral', edgecolor='darkred', linewidth=2)
        elif fp_region.geom_type == 'MultiPolygon':
            for poly in fp_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='lightcoral', edgecolor='darkred', linewidth=2)

    # 绘制FN区域 (漏检 - 黄色)
    if fn_region and not fn_region.is_empty:
        if fn_region.geom_type == 'Polygon':
            x, y = fn_region.exterior.xy
            ax3.fill(x, y, alpha=0.7, facecolor='yellow', edgecolor='orange', linewidth=2)
        elif fn_region.geom_type == 'MultiPolygon':
            for poly in fn_region.geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, facecolor='yellow', edgecolor='orange', linewidth=2)

    # 评估指标统计
    if eval_metrics and direction in eval_metrics.get('results', {}):
        metrics = eval_metrics['results'][direction]
        grid_metrics = metrics.get('grid_metrics', {})

        stats_text = (
            f"TP: {tp_area:.0f} km·min\n"
            f"FP: {fp_area:.0f} km·min\n"
            f"FN: {fn_area:.0f} km·min\n"
            f"━━━━━━━━━━\n"
            f"IoU: {metrics.get('polygon_iou', 0):.3f}\n"
            f"F1: {grid_metrics.get('f1_score', 0):.3f}\n"
            f"Prec: {grid_metrics.get('precision', 0):.3f}\n"
            f"Recall: {grid_metrics.get('recall', 0):.3f}"
        )
    else:
        stats_text = (
            f"TP: {tp_area:.0f} km·min\n"
            f"FP: {fp_area:.0f} km·min\n"
            f"FN: {fn_area:.0f} km·min\n"
            f"IoU: {tp_area/total_area:.3f}" if total_area > 0 else "No metrics"
        )

    ax3.text(0.02, 0.98, stats_text,
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'),
             family='monospace')

    # 统一坐标轴范围
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

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', alpha=0.7, label='TP: True Positive (正确预测)'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', alpha=0.7, label='FP: False Positive (误报)'),
        mpatches.Patch(facecolor='yellow', edgecolor='orange', alpha=0.7, label='FN: False Negative (漏检)')
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
    print("🔍 Enhanced Evaluation Visualization")
    print("   Reading from existing workflow results...")
    print("="*70)

    # 路径配置
    state_path = "output/workflow_state.json"
    eval_dir = "output/evaluation"
    output_dir = "output/enhanced_eval"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载workflow state
    if not os.path.exists(state_path):
        print(f"❌ 找不到文件: {state_path}")
        return

    print(f"\n📂 Loading: {state_path}")
    state = load_workflow_state(state_path)

    # 提取基本信息
    route = state.get('route', '関越道')
    direction_data = state.get('direction_data', {})
    llm_analysis = state.get('llm_analysis', {})

    if not direction_data or not llm_analysis:
        print("❌ workflow_state.json 缺少 direction_data 或 llm_analysis")
        return

    print(f"   Route: {route}")
    print(f"   Directions: {list(direction_data.keys())}")

    # 为每个方向创建可视化
    generated_files = []

    for direction in direction_data.keys():
        print(f"\n{'─'*70}")
        print(f"📊 Processing direction: {direction}")
        print(f"{'─'*70}")

        if direction not in llm_analysis:
            print(f"   ⚠️  跳过：llm_analysis 中没有 {direction} 方向的数据")
            continue

        # 获取数据
        gt_triangles = direction_data[direction].get('triangles', [])
        pred_hotspots = llm_analysis[direction].get('hotspots', [])

        # 加载评估结果
        eval_metrics = load_evaluation_results(eval_dir, route, direction)

        if not eval_metrics:
            print(f"   ⚠️  未找到评估文件，将不显示评估指标")
        else:
            print(f"   ✓ 已加载评估指标")

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"enhanced_eval_{route}_{direction}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 创建可视化
        try:
            result_path = create_enhanced_comparison(
                direction=direction,
                gt_triangles=gt_triangles,
                pred_hotspots=pred_hotspots,
                eval_metrics=eval_metrics,
                output_path=output_path,
                road_name=route
            )
            generated_files.append(result_path)
        except Exception as e:
            print(f"   ❌ 创建可视化失败: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "="*70)
    print("✅ 增强评估可视化完成！")
    print("="*70)
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 生成文件数: {len(generated_files)}")
    for f in generated_files:
        print(f"   • {os.path.basename(f)}")
    print("="*70)


if __name__ == "__main__":
    main()
