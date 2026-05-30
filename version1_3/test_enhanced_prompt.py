#!/usr/bin/env python3
"""
测试增强版prompt的效果
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from implementation import chatbot, visualization, analyze_with_llm, report
from datetime import datetime

def test_enhanced_prompt():
    """测试增强版prompt"""
    print("=" * 80)
    print("🧪 测试增强版Prompt (含交通专家先验知识)")
    print("=" * 80)

    # 初始化状态
    state = {}

    # 测试查询
    test_query = "2025/5/3の関越の交通状況教えて"
    print(f"\n📝 测试查询: {test_query}")

    # 步骤1: Chatbot解析
    print("\n" + "=" * 80)
    print("步骤1: Chatbot解析查询")
    print("=" * 80)
    state['user_input'] = test_query
    state = chatbot(state)

    if state.get('error'):
        print(f"❌ Chatbot错误: {state['error']}")
        return False

    print(f"✅ 解析成功:")
    print(f"  - 年份: {state.get('year', 'N/A')}")
    print(f"  - 月日: {state.get('month', 'N/A')}/{state.get('day', 'N/A')}")
    print(f"  - 路线: {state.get('route', 'N/A')}")

    # 步骤2: 可视化和数据加载
    print("\n" + "=" * 80)
    print("步骤2: 加载历史数据")
    print("=" * 80)
    state = visualization(state)

    if state.get('error'):
        print(f"❌ 可视化错误: {state['error']}")
        return False

    direction_data = state.get('direction_data', {})
    print(f"✅ 数据加载成功:")
    for direction, data in direction_data.items():
        triangle_count = len(data.get('triangles', []))
        print(f"  - {direction}方向: {triangle_count}个三角形")

    # 步骤3: LLM分析（重点测试）
    print("\n" + "=" * 80)
    print("步骤3: LLM分析 (使用增强版Prompt)")
    print("=" * 80)

    state = analyze_with_llm(state)

    if state.get('error'):
        print(f"❌ LLM分析错误: {state['error']}")
        return False

    # 检查LLM分析结果
    llm_analysis = state.get('llm_analysis', {})

    print(f"\n✅ LLM分析完成:")
    for direction, analysis in llm_analysis.items():
        hotspots = analysis.get('hotspots', [])
        reasoning_summary = analysis.get('reasoning_summary', '')

        print(f"\n  {direction}方向:")
        print(f"    识别热点数量: {len(hotspots)}")
        print(f"    推理总结: {reasoning_summary[:100]}...")

        # 检查新字段
        for i, hotspot in enumerate(hotspots[:3], 1):  # 只显示前3个
            print(f"\n    热点 #{i}:")
            print(f"      - ID: {hotspot.get('hotspot_id')}")
            print(f"      - 严重程度: {hotspot.get('severity')}")
            print(f"      - 年份覆盖: {hotspot.get('years_coverage', [])} (重点检查!)")

            # 检查预测形状
            prediction_shape = hotspot.get('prediction_shape', {})
            if prediction_shape:
                shape_type = prediction_shape.get('shape_type', 'unknown')
                vertices = prediction_shape.get('vertices', [])
                print(f"      - 形状类型: {shape_type} (✅ 新格式)")
                print(f"      - 顶点数量: {len(vertices)}")
                print(f"      - 顶点坐标: {vertices}")
            else:
                print(f"      - ⚠️ 使用旧格式 (无prediction_shape)")

            # 检查推理过程
            reasoning = hotspot.get('reasoning', '')
            if reasoning:
                print(f"      - 推理: {reasoning[:80]}...")
            else:
                print(f"      - ⚠️ 缺少推理字段")

    # 检查是否有梯形预测
    has_trapezoid = False
    for direction, analysis in llm_analysis.items():
        for hotspot in analysis.get('hotspots', []):
            prediction_shape = hotspot.get('prediction_shape', {})
            if prediction_shape.get('shape_type') == 'trapezoid':
                has_trapezoid = True
                break

    if has_trapezoid:
        print("\n  ✅ 发现梯形预测！LLM正确使用了新格式")
    else:
        print("\n  ℹ️ 本次分析未产生梯形预测（可能数据特征不适合）")

    # 步骤4: 生成报告
    print("\n" + "=" * 80)
    print("步骤4: 生成最终报告")
    print("=" * 80)

    state = report(state)

    if state.get('error'):
        print(f"❌ 报告生成错误: {state['error']}")
        return False

    final_response = state.get('final_response', '')
    print(f"✅ 报告生成成功")
    print(f"\n最终响应 (前200字符):")
    print(final_response[:200])

    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)

    checks = {
        "LLM分析成功": 'llm_analysis' in state and len(state['llm_analysis']) > 0,
        "包含years_coverage字段": any(
            'years_coverage' in h
            for d in llm_analysis.values()
            for h in d.get('hotspots', [])
        ),
        "包含prediction_shape字段": any(
            'prediction_shape' in h
            for d in llm_analysis.values()
            for h in d.get('hotspots', [])
        ),
        "包含reasoning字段": any(
            'reasoning' in h
            for d in llm_analysis.values()
            for h in d.get('hotspots', [])
        ),
        "生成可视化": 'comparison_paths' in state,
        "生成最终报告": 'final_response' in state
    }

    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    all_passed = all(checks.values())

    if all_passed:
        print("\n🎉 所有检查通过！增强版prompt工作正常。")
        return True
    else:
        print("\n⚠️ 部分检查未通过，请查看详细日志。")
        return False


if __name__ == "__main__":
    success = test_enhanced_prompt()
    sys.exit(0 if success else 1)
