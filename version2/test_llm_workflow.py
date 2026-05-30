#!/usr/bin/env python3
"""
测试脚本 - 测试LLM分析工作流并进行评估
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from implementation import compiled_agent
from config import config
from evaluation import Evaluator
from functions import process_direction_aware_traffic_data


def load_ground_truth(ground_truth_file: str, direction: str) -> dict:
    """
    加载真实数据作为ground truth

    Args:
        ground_truth_file: 真实数据CSV文件路径
        direction: 方向 ("上" 或 "下")

    Returns:
        格式化的ground truth数据 {"上": [...], "下": [...]}
    """
    if not os.path.exists(ground_truth_file):
        print(f"⚠️  Ground truth文件不存在: {ground_truth_file}")
        return {}

    print(f"\n📂 加载Ground Truth数据: {ground_truth_file}")

    try:
        # 使用现有函数处理真实数据
        triangles_data, _ = process_direction_aware_traffic_data(
            file_path=ground_truth_file,
            direction=direction,
            output_dir="output/temp"
        )

        # 转换为标准格式
        triangles = []
        for t in triangles_data:
            vertices = t.get('vertices', [])
            if len(vertices) >= 3:
                triangles.append({
                    'vertices': vertices,
                    'kp_start': t.get('kp_start'),
                    'kp_end': t.get('kp_end'),
                    'time_start': t.get('time_start'),
                    'time_end': t.get('time_end'),
                    'time_peak': t.get('time_peak'),
                    'area': t.get('area')
                })

        print(f"✓ Ground Truth加载完成: {len(triangles)} 个拥堵事件")
        return {direction: triangles}

    except Exception as e:
        print(f"❌ 加载Ground Truth失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_evaluation(predictions: dict, ground_truth: dict, route: str, direction: str):
    """
    运行评估

    Args:
        predictions: 预测结果 {"上": [...], "下": [...]}
        ground_truth: 真实数据 {"上": [...], "下": [...]}
        route: 道路名称
        direction: 方向
    """
    print("\n" + "="*70)
    print("📊 开始评估预测结果")
    print("="*70)

    # 转换预测数据格式
    pred_formatted = {}
    for dir_key, data in predictions.items():
        if isinstance(data, dict) and 'hotspots' in data:
            pred_formatted[dir_key] = data['hotspots']
        else:
            pred_formatted[dir_key] = data

    # 创建评估器
    evaluator = Evaluator(
        road_type=route,
        direction=direction,
        time_step_minutes=60
    )

    # 执行评估
    results = evaluator.evaluate_all_directions(pred_formatted, ground_truth)

    # 打印汇总报告
    evaluator.print_summary_report(results)

    # 保存评估结果
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output = os.path.join(output_dir, f"evaluation_{route}_{direction}_{timestamp}.json")
    evaluator.save_results_to_json(results, eval_output)

    return results


def main(ground_truth_file: str = None):
    """
    主函数

    Args:
        ground_truth_file: 可选的ground truth数据文件路径
    """
    print("="*70)
    print("🧪 Testing LLM-based Workflow with Evaluation")
    print("="*70)

    # 检查配置
    print("\n📋 Configuration:")
    print(f"   API Key: {'Set ✅' if config.get_openai_api_key() else 'Not set ⚠️'}")
    print(f"   Model: {config.openai_model}")
    print(f"   LLM Available: {config.is_llm_available()}")

    if not config.is_llm_available():
        print("\n⚠️  Warning: No API key configured, will use fallback analysis")

    # 测试输入
    test_input = "请分析関越高速公路下行方向2024年5月3日的交通情况"

    print(f"\n📝 Test Input:")
    print(f"   {test_input}")

    print("\n" + "="*70)
    print("🚀 Running workflow...")
    print("="*70)

    try:
        # 执行工作流
        result = compiled_agent.invoke({"user_input": test_input})

        print("\n" + "="*70)
        print("✅ Workflow completed successfully!")
        print("="*70)

        # 显示结果
        print("\n📊 Prediction Results:")

        # 提取关键信息
        route = result.get('route', '関越道')
        direction = result.get('direction', '下')

        # LLM分析结果
        llm_analysis = result.get('llm_analysis', {})
        if llm_analysis:
            print(f"\n🤖 LLM Analysis:")
            for dir_key, analysis in llm_analysis.items():
                hotspots = analysis.get('hotspots', [])
                summary = analysis.get('summary', {})
                confidence = summary.get('analysis_confidence', 0)

                print(f"\n   {dir_key} direction:")
                print(f"      Hotspots: {len(hotspots)}")
                print(f"      Confidence: {confidence:.2f}")

                for i, hotspot in enumerate(hotspots[:3], 1):
                    kp_range = hotspot.get('kp_range', [0, 0])
                    freq = hotspot.get('frequency', 0)
                    print(f"      Hotspot {i}: KP {kp_range[0]:.1f}-{kp_range[1]:.1f}km, "
                          f"frequency: {freq}")

        # 文件输出
        print(f"\n📁 Generated Files:")

        fig_paths = result.get('fig_paths', [])
        if fig_paths:
            for i, path in enumerate(fig_paths, 1):
                if os.path.exists(path):
                    print(f"   🖼️  Image {i}: {path}")

        csv_path = result.get('csv_output_path', '')
        if csv_path and os.path.exists(csv_path):
            print(f"   📄 CSV: {csv_path}")

        # 报告预览
        final_report = result.get('final_report', '')
        if final_report:
            print(f"\n📄 Report Preview (first 300 chars):")
            print(f"   {final_report[:300]}...")

        # ==================== 评估部分 ====================
        if ground_truth_file:
            # 加载ground truth
            ground_truth = load_ground_truth(ground_truth_file, direction)

            if ground_truth and llm_analysis:
                # 运行评估
                eval_results = run_evaluation(
                    predictions=llm_analysis,
                    ground_truth=ground_truth,
                    route=route,
                    direction=direction
                )

                # 显示评估摘要
                print("\n" + "="*70)
                print("📈 Evaluation Summary")
                print("="*70)

                for dir_key, metrics in eval_results.items():
                    if dir_key == "average":
                        print(f"\n【整体平均】")
                    else:
                        print(f"\n【{dir_key}行】")

                    print(f"  多边形IoU:      {metrics['polygon_iou']:.4f}")
                    print(f"  F1-Score:       {metrics['grid_metrics']['f1_score']:.4f}")
                    print(f"  Precision:      {metrics['grid_metrics']['precision']:.4f}")
                    print(f"  Recall:         {metrics['grid_metrics']['recall']:.4f}")
            else:
                print("\n⚠️  无法进行评估：Ground truth或预测结果为空")
        else:
            print("\n💡 未提供Ground Truth文件，跳过评估")
            print("   提示：使用 --gt 参数指定ground truth文件")

        print("\n" + "="*70)
        print("🎉 Test completed successfully!")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试LLM工作流并评估')
    parser.add_argument(
        '--gt',
        type=str,
        default=None,
        help='Ground truth CSV文件路径（可选）'
    )

    args = parser.parse_args()

    success = main(ground_truth_file=args.gt)
    sys.exit(0 if success else 1)
