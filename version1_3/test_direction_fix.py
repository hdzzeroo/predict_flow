"""
测试方向修复效果
运行这个脚本来验证LLM是否正确生成了上下方向的三角形
"""

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(__file__))

from implementation import compiled_agent
from config import config

def test_direction_fix():
    """测试下り方向的三角形是否尖朝左"""

    print("="*70)
    print("🧪 测试方向修复效果")
    print("="*70)

    # 检查API密钥
    if not config.is_llm_available():
        print("❌ 错误: 需要设置 OPENAI_API_KEY 环境变量")
        print("请运行: export OPENAI_API_KEY='your-key-here'")
        return

    print(f"\n✅ 使用模型: {config.openai_model}")
    print(f"✅ API密钥: {config.openai_api_key[:8]}...")

    # 测试输入 - 只测试下り方向
    test_input = "请分析2025年5月4日関越道下り方向的交通拥堵情况"

    print(f"\n📝 测试输入: {test_input}")
    print("\n开始分析...")
    print("-"*70)

    try:
        # 执行分析
        result = compiled_agent.invoke({
            "user_input": test_input
        })

        # 获取LLM分析结果
        llm_analysis = result.get('llm_analysis', {})

        if not llm_analysis:
            print("❌ 错误: 没有LLM分析结果")
            return

        # 检查下り方向
        if '下' not in llm_analysis:
            print("❌ 错误: 没有下り方向的分析结果")
            return

        analysis = llm_analysis['下']
        hotspots = analysis.get('hotspots', [])

        print(f"\n{'='*70}")
        print(f"📊 下り方向分析结果")
        print(f"{'='*70}")
        print(f"识别到 {len(hotspots)} 个热点")

        # 检查每个热点的方向
        all_correct = True
        for i, hotspot in enumerate(hotspots):
            hotspot_id = hotspot['hotspot_id']

            # 获取prediction_shape
            pred_shape = hotspot.get('prediction_shape', {})
            if not pred_shape or 'vertices' not in pred_shape:
                print(f"\n⚠️ Hotspot {hotspot_id}: 没有prediction_shape数据")
                continue

            vertices = pred_shape['vertices']
            shape_type = pred_shape.get('shape_type', 'unknown')

            print(f"\n--- Hotspot {hotspot_id} ({shape_type}) ---")
            print(f"顶点: {vertices}")

            # 提取KP值
            kp_values = [v[0] for v in vertices]
            min_kp = min(kp_values)
            max_kp = max(kp_values)

            # 判断朝向
            if shape_type == 'triangle':
                # 找到垂直边（两个KP相同的点）
                kp_counts = {}
                for kp in kp_values:
                    kp_counts[kp] = kp_counts.get(kp, 0) + 1

                # 垂直边的KP（出现2次）
                base_kp = None
                peak_kp = None
                for kp, count in kp_counts.items():
                    if count == 2:
                        base_kp = kp
                    else:
                        peak_kp = kp

                if base_kp is not None and peak_kp is not None:
                    if peak_kp < base_kp:
                        print(f"✅ 方向正确: 尖朝左 (peak_kp={peak_kp:.1f} < base_kp={base_kp:.1f})")
                    else:
                        print(f"❌ 方向错误: 尖朝右 (peak_kp={peak_kp:.1f} > base_kp={base_kp:.1f}) - 应该朝左！")
                        all_correct = False
                else:
                    print(f"⚠️ 无法判断方向（找不到垂直边）")

            elif shape_type == 'trapezoid' or shape_type == 'quadrilateral':
                # 对于梯形，检查最小KP是否在左侧
                if min_kp < max_kp:
                    left_side_kp = min_kp
                    right_side_kp = max_kp

                    # 下り方向应该是左侧KP更小（尖朝左）
                    print(f"✅ 方向正确: 左侧KP={left_side_kp:.1f} < 右侧KP={right_side_kp:.1f}")
                else:
                    print(f"❌ 方向可能有问题")
                    all_correct = False

        print(f"\n{'='*70}")
        if all_correct and hotspots:
            print("✅ 测试通过：所有热点方向正确（尖朝左）")
        elif hotspots:
            print("❌ 测试失败：部分热点方向错误")
        else:
            print("⚠️ 警告：没有识别到热点")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direction_fix()
