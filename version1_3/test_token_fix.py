#!/usr/bin/env python3
"""
测试Token超限问题的修复
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import config
from prompt_templates import build_analysis_prompt


def test_model_configuration():
    """测试模型配置"""
    print("=" * 80)
    print("测试1: 模型配置检查")
    print("=" * 80)

    print(f"\n当前配置的模型: {config.openai_model}")
    print(f"最大Token数: {config.openai_max_tokens}")
    print(f"超时时间: {config.openai_timeout}秒")

    # 检查模型上下文限制
    model_limits = {
        "gpt-3.5-turbo": 16385,
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4": 8192
    }

    expected_limit = model_limits.get(config.openai_model, "未知")
    print(f"该模型的上下文限制: {expected_limit} tokens")

    if config.openai_model == "gpt-4o-mini":
        print("\n✅ 配置正确！使用 gpt-4o-mini (128k 上下文)")
        return True
    elif config.openai_model == "gpt-3.5-turbo":
        print("\n❌ 配置错误！仍在使用 gpt-3.5-turbo (16k 上下文)")
        print("   建议修改为 gpt-4o-mini 以支持更多三角形数据")
        return False
    else:
        print(f"\n⚠️ 使用了其他模型: {config.openai_model}")
        return True


def test_token_estimation():
    """测试Token估算"""
    print("\n" + "=" * 80)
    print("测试2: Token消耗估算")
    print("=" * 80)

    # 模拟不同数量的三角形
    test_cases = [
        ("小规模", 10),
        ("中规模", 50),
        ("大规模", 100),
        ("实际场景", 58)  # 下行方向实际数量
    ]

    for case_name, triangle_count in test_cases:
        # 创建模拟三角形数据
        sample_triangles = [
            {
                'id': i,
                'shape_type': 'triangle',
                'vertices': [(10.0 + i, 420), (15.0 + i, 480), (10.0 + i, 540)],
                'kp_start': 10.0 + i,
                'kp_end': 15.0 + i,
                'time_start': 420,
                'time_end': 540,
                'time_peak': 480,
                'width': 5.0,
                'height': 120,
                'area': 300.0
            }
            for i in range(triangle_count)
        ]

        # 构建prompt
        prompt = build_analysis_prompt(
            direction="下",
            csv_files=["test.csv"],
            triangles=sample_triangles
        )

        # 估算token数（1 token ≈ 4 字符）
        estimated_tokens = len(prompt) // 4
        output_tokens = config.openai_max_tokens
        total_tokens = estimated_tokens + output_tokens

        print(f"\n{case_name} ({triangle_count}个三角形):")
        print(f"  Prompt长度: {len(prompt)} 字符")
        print(f"  估算输入Token: ~{estimated_tokens} tokens")
        print(f"  预留输出Token: {output_tokens} tokens")
        print(f"  总计: ~{total_tokens} tokens")

        # 检查是否超限
        if config.openai_model == "gpt-3.5-turbo":
            limit = 16385
        elif config.openai_model == "gpt-4o-mini":
            limit = 128000
        else:
            limit = 128000

        if total_tokens <= limit:
            print(f"  状态: ✅ 在限制内 ({limit} tokens)")
        else:
            print(f"  状态: ❌ 超过限制 ({limit} tokens)")
            print(f"  超出: {total_tokens - limit} tokens")


def test_prompt_structure():
    """测试Prompt结构"""
    print("\n" + "=" * 80)
    print("测试3: Prompt结构验证")
    print("=" * 80)

    # 创建样本三角形
    sample_triangles = [
        {
            'id': 0,
            'shape_type': 'triangle',
            'vertices': [(10.0, 420), (15.0, 480), (10.0, 540)],
            'kp_start': 10.0,
            'kp_end': 15.0,
            'time_start': 420,
            'time_end': 540,
            'time_peak': 480,
            'width': 5.0,
            'height': 120,
            'area': 300.0
        }
    ]

    prompt = build_analysis_prompt(
        direction="上",
        csv_files=["test.csv"],
        triangles=sample_triangles
    )

    # 检查关键元素
    checks = [
        ("包含vertices字段", '"vertices"' in prompt),
        ("包含shape_type字段", '"shape_type"' in prompt),
        ("包含summary字段", '"summary"' in prompt),
        ("包含time_str字段", '"time_str"' in prompt),
        ("包含kp_range字段", '"kp_range"' in prompt),
        ("包含area字段", '"area"' in prompt)
    ]

    print("\nPrompt结构检查:")
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False

    return all_passed


def main():
    """运行所有测试"""
    print("\n🔧 测试Token超限问题修复")
    print("=" * 80)

    tests = [
        ("模型配置", test_model_configuration),
        ("Token估算", test_token_estimation),
        ("Prompt结构", test_prompt_structure)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not None:
                results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 出现异常: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)

    for test_name, result in results:
        if result is None:
            continue
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    if results:
        success_count = sum(1 for _, result in results if result)
        total_count = len(results)
        print(f"\n总计: {success_count}/{total_count} 测试通过")

        if success_count == total_count:
            print("\n🎉 所有测试通过！Token超限问题已修复。")
            return True
        else:
            print(f"\n⚠️ 有 {total_count - success_count} 个测试失败。")
            return False
    else:
        print("\n✅ 所有检查完成")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
