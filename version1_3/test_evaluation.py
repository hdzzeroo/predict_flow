"""
评估功能测试脚本
用于测试evaluation模块的功能
"""

import json
import os
import pandas as pd
from evaluation import Evaluator
from functions import process_direction_aware_traffic_data


def test_evaluation_with_sample_data():
    """使用示例数据测试评估功能"""

    print("="*70)
    print("评估功能测试")
    print("="*70)

    # 创建模拟的预测数据
    print("\n【步骤1】创建模拟预测数据...")

    predictions = {
        "下": [
            {
                "hotspot_id": 1,
                "prediction_shape": {
                    "shape_type": "triangle",
                    "vertices": [
                        [29.0, 480],   # KP 29km, 8:00 (480分钟)
                        [29.0, 600],   # KP 29km, 10:00 (600分钟)
                        [35.0, 540]    # KP 35km, 9:00 (540分钟)
                    ]
                },
                "kp_range": [29.0, 35.0],
                "time_range": [480, 600]
            },
            {
                "hotspot_id": 2,
                "prediction_shape": {
                    "shape_type": "triangle",
                    "vertices": [
                        [4.0, 420],    # KP 4km, 7:00 (420分钟)
                        [4.0, 480],    # KP 4km, 8:00 (480分钟)
                        [8.0, 450]     # KP 8km, 7:30 (450分钟)
                    ]
                },
                "kp_range": [4.0, 8.0],
                "time_range": [420, 480]
            }
        ]
    }

    print(f"  - 预测热点数量: {len(predictions['下'])}")

    # 创建模拟的真实数据
    print("\n【步骤2】创建模拟真实数据...")

    ground_truth = {
        "下": [
            {
                "vertices": [
                    [30.0, 490],
                    [30.0, 580],
                    [34.0, 535]
                ],
                "kp_start": 30.0,
                "kp_end": 34.0,
                "time_start": 490,
                "time_end": 580,
                "time_peak": 535
            },
            {
                "vertices": [
                    [31.0, 500],
                    [31.0, 560],
                    [33.0, 530]
                ],
                "kp_start": 31.0,
                "kp_end": 33.0,
                "time_start": 500,
                "time_end": 560,
                "time_peak": 530
            },
            {
                "vertices": [
                    [4.5, 430],
                    [4.5, 470],
                    [7.0, 450]
                ],
                "kp_start": 4.5,
                "kp_end": 7.0,
                "time_start": 430,
                "time_end": 470,
                "time_peak": 450
            }
        ]
    }

    print(f"  - 真实拥堵数量: {len(ground_truth['下'])}")

    # 执行评估
    print("\n【步骤3】执行评估...")
    print("-"*70)

    evaluator = Evaluator(road_type="関越道", direction="下", time_step_minutes=60)
    results = evaluator.evaluate_all_directions(predictions, ground_truth)

    # 打印汇总报告
    print("\n" + "="*70)
    evaluator.print_summary_report(results)
    print("="*70)

    # 保存结果
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "test_evaluation_result.json")
    evaluator.save_results_to_json(results, output_path)

    print("\n✓ 测试完成！")

    return results


def test_evaluation_with_real_data():
    """使用真实的2025年数据测试评估功能"""

    print("="*70)
    print("使用真实数据测试评估")
    print("="*70)

    # 检查真实数据文件是否存在
    gt_file = "data/processed_data/関越道_下_2025_05-05.csv"

    if not os.path.exists(gt_file):
        print(f"❌ 真实数据文件不存在: {gt_file}")
        print("请确保文件路径正确，或使用 test_evaluation_with_sample_data() 测试")
        return None

    # 加载真实数据
    print(f"\n【步骤1】加载真实数据: {gt_file}")

    df = pd.read_csv(gt_file, encoding='utf-8')
    print(f"  - 数据行数: {len(df)}")
    print(f"  - 列名: {df.columns.tolist()}")

    # 处理真实数据
    print("\n【步骤2】处理真实数据生成三角形...")

    try:
        triangles = process_direction_aware_traffic_data(
            df=df,
            direction="下",
            route_name="関越道",
            year=2025
        )

        print(f"  ✓ 成功生成 {len(triangles)} 个三角形")

        # 转换为评估格式
        gt_formatted = []
        for t in triangles:
            if 'vertices' in t and len(t['vertices']) >= 3:
                gt_formatted.append({
                    'vertices': t['vertices'],
                    'kp_start': t.get('kp_start'),
                    'kp_end': t.get('kp_end'),
                    'time_start': t.get('time_start'),
                    'time_end': t.get('time_end'),
                    'time_peak': t.get('time_peak')
                })

        ground_truth = {"下": gt_formatted}
        print(f"  ✓ 格式化后: {len(gt_formatted)} 个有效三角形")

    except Exception as e:
        print(f"  ❌ 处理真实数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 创建示例预测数据（基于真实数据的大致范围）
    print("\n【步骤3】创建示例预测数据...")

    # 分析真实数据的KP和时间范围
    if gt_formatted:
        kp_values = []
        time_values = []
        for t in gt_formatted:
            vertices = t['vertices']
            for v in vertices:
                kp_values.append(v[0])
                time_values.append(v[1])

        kp_min, kp_max = min(kp_values), max(kp_values)
        time_min, time_max = min(time_values), max(time_values)

        print(f"  - KP范围: [{kp_min:.1f}, {kp_max:.1f}] km")
        print(f"  - 时间范围: [{time_min:.0f}, {time_max:.0f}] min ({time_min//60:02.0f}:{time_min%60:02.0f} - {time_max//60:02.0f}:{time_max%60:02.0f})")

        # 创建覆盖主要区域的预测热点
        predictions = {
            "下": [
                {
                    "hotspot_id": 1,
                    "prediction_shape": {
                        "shape_type": "trapezoid",
                        "vertices": [
                            [kp_min, time_min + 30],
                            [kp_max, time_min],
                            [kp_max, time_max],
                            [kp_min, time_max - 30]
                        ]
                    }
                }
            ]
        }

        print(f"  - 创建 {len(predictions['下'])} 个预测热点")
    else:
        print("  ⚠️ 真实数据为空，无法创建示例预测")
        return None

    # 执行评估
    print("\n【步骤4】执行评估...")
    print("-"*70)

    evaluator = Evaluator(road_type="関越道", direction="下", time_step_minutes=60)
    results = evaluator.evaluate_all_directions(predictions, ground_truth)

    # 打印汇总报告
    print("\n" + "="*70)
    evaluator.print_summary_report(results)
    print("="*70)

    # 保存结果
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "test_real_data_evaluation.json")
    evaluator.save_results_to_json(results, output_path)

    print("\n✓ 测试完成！")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试评估功能')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sample', 'real'],
        default='sample',
        help='测试模式: sample=使用模拟数据, real=使用真实数据'
    )

    args = parser.parse_args()

    if args.mode == 'sample':
        print("使用模拟数据测试...\n")
        test_evaluation_with_sample_data()
    else:
        print("使用真实数据测试...\n")
        test_evaluation_with_real_data()


if __name__ == "__main__":
    main()
