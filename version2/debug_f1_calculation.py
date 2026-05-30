#!/usr/bin/env python3
"""
调试F1-Score计算
详细展示栅格化过程和混淆矩阵
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import json
import pandas as pd
from functions import process_direction_aware_traffic_data


def load_kp_intervals(road_type: str = "関越道", direction: str = "下"):
    """加载KP区间"""
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data"
    csv_file = f"{base_dir}/roadic_kannetsu.csv"

    df = pd.read_csv(csv_file)
    direction_map = {"上": "up", "下": "down"}
    dir_en = direction_map.get(direction, "down")
    df_filtered = df[df['direction'] == dir_en].copy()

    kp_values = sorted(df_filtered['KP'].unique())
    kp_intervals = []
    for i in range(len(kp_values) - 1):
        kp_intervals.append((kp_values[i], kp_values[i+1]))

    return kp_intervals


def rasterize_polygons(polygons, kp_intervals, time_min, time_max, time_step):
    """栅格化多边形"""
    n_kp = len(kp_intervals)
    n_time = int(np.ceil((time_max - time_min) / time_step))
    grid = np.zeros((n_kp, n_time), dtype=np.uint8)

    union_poly = unary_union(polygons)

    for i, (kp_start, kp_end) in enumerate(kp_intervals):
        for j in range(n_time):
            t_start = time_min + j * time_step
            t_end = time_min + (j + 1) * time_step

            grid_rect = box(kp_start, t_start, kp_end, t_end)

            # 关键：只要有相交就标记为1
            if union_poly.intersects(grid_rect):
                grid[i, j] = 1

    return grid, n_time


def visualize_grid(pred_grid, gt_grid, kp_intervals, time_min, time_max, time_step, direction):
    """可视化栅格和混淆矩阵"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图1: Ground Truth栅格
    ax1 = axes[0, 0]
    im1 = ax1.imshow(gt_grid, aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_title(f'Ground Truth Grid (2025年真实)\n{direction}方向', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step (每格={time_step}min)')
    ax1.set_ylabel('KP Interval Index')
    plt.colorbar(im1, ax=ax1)

    # 子图2: Prediction栅格
    ax2 = axes[0, 1]
    im2 = ax2.imshow(pred_grid, aspect='auto', cmap='Reds', interpolation='nearest')
    ax2.set_title(f'Prediction Grid (LLM预测)\n{direction}方向', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step (每格={time_step}min)')
    ax2.set_ylabel('KP Interval Index')
    plt.colorbar(im2, ax=ax2)

    # 子图3: 重叠分析
    ax3 = axes[1, 0]
    overlap_grid = np.zeros_like(gt_grid, dtype=np.int8)
    overlap_grid[(pred_grid == 1) & (gt_grid == 1)] = 3  # TP - 绿色
    overlap_grid[(pred_grid == 1) & (gt_grid == 0)] = 2  # FP - 红色
    overlap_grid[(pred_grid == 0) & (gt_grid == 1)] = 1  # FN - 黄色

    from matplotlib.colors import ListedColormap
    colors = ['white', 'yellow', 'red', 'green']
    cmap = ListedColormap(colors)

    im3 = ax3.imshow(overlap_grid, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    ax3.set_title('TP/FP/FN Grid\n绿=TP(正确) 红=FP(误报) 黄=FN(漏检)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('KP Interval Index')

    # 子图4: 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    tp = np.sum((pred_grid == 1) & (gt_grid == 1))
    fp = np.sum((pred_grid == 1) & (gt_grid == 0))
    fn = np.sum((pred_grid == 0) & (gt_grid == 1))
    tn = np.sum((pred_grid == 0) & (gt_grid == 0))

    total_cells = pred_grid.size
    gt_positive = np.sum(gt_grid == 1)
    pred_positive = np.sum(pred_grid == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    stats_text = f"""
栅格化评估详细统计

━━━━━━━━━━━━━━━━━━━━━━━━
【栅格信息】
  总栅格数: {total_cells}
  KP区间: {len(kp_intervals)}
  时间段: {pred_grid.shape[1]}
  时间粒度: {time_step} 分钟

━━━━━━━━━━━━━━━━━━━━━━━━
【Ground Truth】
  有拥堵的栅格: {gt_positive} ({gt_positive/total_cells*100:.1f}%)
  无拥堵的栅格: {total_cells-gt_positive} ({(total_cells-gt_positive)/total_cells*100:.1f}%)

【Prediction】
  预测有拥堵: {pred_positive} ({pred_positive/total_cells*100:.1f}%)
  预测无拥堵: {total_cells-pred_positive} ({(total_cells-pred_positive)/total_cells*100:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━
【混淆矩阵】
  TP (预测对,实际对): {tp} ({tp/total_cells*100:.1f}%)
  FP (预测对,实际错): {fp} ({fp/total_cells*100:.1f}%)
  FN (预测错,实际对): {fn} ({fn/total_cells*100:.1f}%)
  TN (预测错,实际错): {tn} ({tn/total_cells*100:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━
【评估指标】
  Precision: {precision:.4f}
    = TP/(TP+FP) = {tp}/{tp+fp}
    含义: 预测为拥堵的格子中，{precision*100:.1f}%确实有拥堵

  Recall: {recall:.4f}
    = TP/(TP+FN) = {tp}/{tp+fn}
    含义: 真实拥堵的格子中，{recall*100:.1f}%被预测到

  F1-Score: {f1:.4f}
    = 2*P*R/(P+R)

━━━━━━━━━━━━━━━━━━━━━━━━
【问题分析】
  预测覆盖率: {pred_positive/gt_positive:.2f}x
    (预测{pred_positive}格 vs 真实{gt_positive}格)

  {'⚠️ 预测范围过大！' if pred_positive > gt_positive*1.5 else '✓ 预测范围合理'}
  {'⚠️ 漏检严重！' if fn > tp*0.5 else '✓ 漏检可接受'}
  {'⚠️ 误报严重！' if fp > tp else '✓ 误报可接受'}
"""

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'output/debug_f1_grid_{direction}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 栅格可视化已保存: output/debug_f1_grid_{direction}.png")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🔍 F1-Score计算调试")
    print("="*70)

    # 加载workflow state
    with open('output/workflow_state.json', 'r') as f:
        state = json.load(f)

    route = state['route']
    target_year = state['target_year']
    target_month = state['target_month']
    target_day = state['target_day']

    # 测试下方向
    direction = "下"

    print(f"\n分析 {direction} 方向")
    print("-"*70)

    # 1. 加载2025年GT
    base_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    gt_file = f"{base_dir}/{route}_{direction}_{target_year}_{target_month:02d}-{target_day:02d}.csv"

    print(f"加载GT: {gt_file}")
    gt_triangles_data, _ = process_direction_aware_traffic_data(gt_file, direction, "output/temp")

    gt_polygons = []
    for t in gt_triangles_data:
        if 'vertices' in t and len(t['vertices']) >= 3:
            poly = Polygon(t['vertices'])
            if poly.is_valid and poly.area > 0:
                gt_polygons.append(poly)

    print(f"  GT多边形: {len(gt_polygons)}")

    # 2. 加载预测
    pred_hotspots = state['llm_analysis'][direction]['hotspots']
    pred_polygons = []
    for h in pred_hotspots:
        if 'prediction_shape' in h and 'vertices' in h['prediction_shape']:
            poly = Polygon(h['prediction_shape']['vertices'])
            if poly.is_valid and poly.area > 0:
                pred_polygons.append(poly)

    print(f"  预测多边形: {len(pred_polygons)}")

    # 3. 栅格化
    kp_intervals = load_kp_intervals(route, direction)
    print(f"  KP区间数: {len(kp_intervals)}")

    # 获取时间范围
    all_polys = gt_polygons + pred_polygons
    all_bounds = [p.bounds for p in all_polys]
    time_min = min(b[1] for b in all_bounds)
    time_max = max(b[3] for b in all_bounds)
    time_step = 60  # 60分钟

    print(f"  时间范围: {time_min:.0f} - {time_max:.0f} min")

    # 栅格化
    gt_grid, n_time = rasterize_polygons(gt_polygons, kp_intervals, time_min, time_max, time_step)
    pred_grid, _ = rasterize_polygons(pred_polygons, kp_intervals, time_min, time_max, time_step)

    print(f"  栅格尺寸: {len(kp_intervals)} × {n_time} = {len(kp_intervals)*n_time} 格子")

    # 4. 计算混淆矩阵
    tp = np.sum((pred_grid == 1) & (gt_grid == 1))
    fp = np.sum((pred_grid == 1) & (gt_grid == 0))
    fn = np.sum((pred_grid == 0) & (gt_grid == 1))
    tn = np.sum((pred_grid == 0) & (gt_grid == 0))

    print(f"\n混淆矩阵:")
    print(f"  TP (都是1): {tp}")
    print(f"  FP (预测1,真实0): {fp}")
    print(f"  FN (预测0,真实1): {fn}")
    print(f"  TN (都是0): {tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n指标:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # 5. 可视化
    visualize_grid(pred_grid, gt_grid, kp_intervals, time_min, time_max, time_step, direction)

    # 6. 问题分析
    print("\n" + "="*70)
    print("🔍 为什么F1看起来高？")
    print("="*70)

    total = len(kp_intervals) * n_time
    gt_positive = np.sum(gt_grid == 1)
    pred_positive = np.sum(pred_grid == 1)

    print(f"\n关键点:")
    print(f"1. 总栅格数很大: {total} 个格子")
    print(f"2. 真实拥堵栅格很少: {gt_positive} ({gt_positive/total*100:.1f}%)")
    print(f"3. 预测拥堵栅格: {pred_positive} ({pred_positive/total*100:.1f}%)")
    print(f"4. TN (都没拥堵): {tn} ({tn/total*100:.1f}%)")

    print(f"\n问题所在:")
    print(f"  - 栅格化方法: 只要多边形与格子 **有任何相交** 就标记为1")
    print(f"  - 这导致: 一个大的预测矩形会覆盖很多格子")
    print(f"  - 结果: 即使面积IoU很低(0.23)，栅格F1也能达到0.49")
    print(f"  - 因为: 大部分格子都是TN(都没拥堵)，被排除在计算之外")

    print(f"\nRecall高的原因:")
    print(f"  - 预测范围 {pred_positive/gt_positive:.1f}x 大于真实")
    print(f"  - 覆盖了大部分真实拥堵格子")
    print(f"  - Recall = {tp}/{gt_positive} = {recall:.3f}")

    print(f"\nPrecision低的原因:")
    print(f"  - 预测了很多不存在的拥堵")
    print(f"  - FP({fp}) 是 TP({tp}) 的 {fp/tp:.1f}倍")
    print(f"  - Precision = {tp}/{pred_positive} = {precision:.3f}")


if __name__ == "__main__":
    main()
