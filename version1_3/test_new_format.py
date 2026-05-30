#!/usr/bin/env python3
"""
测试新的混合格式prompt和完整三角形数据保存
"""

import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from prompt_templates import build_analysis_prompt


def test_build_prompt_with_sample_data():
    """测试使用样本数据构建prompt"""
    print("=" * 80)
    print("测试1: 构建混合格式Prompt")
    print("=" * 80)

    # 创建样本三角形数据（模拟完整的数据结构）
    sample_triangles = [
        {
            'id': 0,
            'shape_type': 'triangle',
            'vertices': [
                (79.8, 495),   # (kp, time_min)
                (83.7, 565),
                (79.8, 635)
            ],
            'center': (81.1, 565.0),
            'area': 429.0,
            'width': 3.9,
            'height': 140,
            'kp_start': 79.8,
            'kp_end': 83.7,
            'time_start': 495,
            'time_end': 635,
            'time_peak': 565
        },
        {
            'id': 1,
            'shape_type': 'quadrilateral',
            'vertices': [
                (10.0, 420),
                (15.0, 420),
                (18.0, 480),
                (10.0, 540)
            ],
            'center': (13.25, 465.0),
            'area': 600.0,
            'width': 8.0,
            'height': 120,
            'kp_start': 10.0,
            'kp_end': 18.0,
            'time_start': 420,
            'time_end': 540,
            'time_peak': 480
        },
        {
            'id': 2,
            'shape_type': 'triangle',
            'vertices': [
                (12.0, 430),
                (16.5, 490),
                (12.0, 550)
            ],
            'center': (13.5, 490.0),
            'area': 270.0,
            'width': 4.5,
            'height': 120,
            'kp_start': 12.0,
            'kp_end': 16.5,
            'time_start': 430,
            'time_end': 550,
            'time_peak': 490
        }
    ]

    # 构建prompt
    prompt = build_analysis_prompt(
        direction="上",
        csv_files=["関越道_上_2024_05-03.csv"],
        triangles=sample_triangles,
        max_triangles=None  # 传输所有三角形
    )

    print(f"\n✅ Prompt构建成功")
    print(f"   Prompt长度: {len(prompt)} 字符")
    print(f"   预估Token数: ~{len(prompt) // 4} tokens")

    # 显示prompt的前1000个字符
    print(f"\n📋 Prompt预览 (前1000字符):")
    print("-" * 80)
    print(prompt[:1000])
    print("...")
    print("-" * 80)

    # 检查关键内容
    checks = [
        ("包含vertices字段", '"vertices"' in prompt),
        ("包含shape_type字段", '"shape_type"' in prompt),
        ("包含summary字段", '"summary"' in prompt),
        ("包含time_str字段", '"time_str"' in prompt),
        ("包含所有三角形", prompt.count('"id":') >= len(sample_triangles))
    ]

    print(f"\n✅ 内容检查:")
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")

    return all(result for _, result in checks)


def test_triangles_data_structure():
    """测试三角形数据结构转换"""
    print("\n" + "=" * 80)
    print("测试2: 三角形数据结构转换")
    print("=" * 80)

    # 模拟一个简单的三角形
    triangle = {
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

    # 测试prompt构建
    prompt = build_analysis_prompt(
        direction="上",
        csv_files=["test.csv"],
        triangles=[triangle]
    )

    # 解析prompt中的JSON数据
    import re
    json_match = re.search(r'```json\n(.*?)\n```', prompt, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
        triangles_data = json.loads(json_str)

        print(f"\n✅ 成功提取并解析三角形JSON数据")
        print(f"\n📊 转换后的数据结构:")
        print(json.dumps(triangles_data[0], indent=2, ensure_ascii=False))

        # 验证数据完整性
        t = triangles_data[0]
        checks = [
            ("id字段存在", 'id' in t),
            ("shape_type字段存在", 'shape_type' in t),
            ("vertices字段存在", 'vertices' in t),
            ("vertices包含3个顶点", len(t.get('vertices', [])) == 3),
            ("每个顶点有kp字段", all('kp' in v for v in t.get('vertices', []))),
            ("每个顶点有time_min字段", all('time_min' in v for v in t.get('vertices', []))),
            ("每个顶点有time_str字段", all('time_str' in v for v in t.get('vertices', []))),
            ("summary字段存在", 'summary' in t),
            ("summary包含kp_range", 'kp_range' in t.get('summary', {})),
            ("summary包含area", 'area' in t.get('summary', {}))
        ]

        print(f"\n✅ 数据完整性检查:")
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}: {result}")

        return all(result for _, result in checks)
    else:
        print("❌ 无法从prompt中提取JSON数据")
        return False


def test_backward_compatibility():
    """测试向后兼容性（没有vertices字段的情况）"""
    print("\n" + "=" * 80)
    print("测试3: 向后兼容性测试")
    print("=" * 80)

    # 创建没有vertices字段的旧格式三角形
    old_format_triangle = {
        'id': 0,
        'kp_start': 10.0,
        'kp_end': 15.0,
        'time_start': 420,
        'time_end': 540,
        'time_peak': 480,
        'area': 300.0
    }

    print(f"\n📝 测试旧格式三角形（无vertices字段）")

    try:
        prompt = build_analysis_prompt(
            direction="上",
            csv_files=["test.csv"],
            triangles=[old_format_triangle]
        )

        # 检查是否自动构造了vertices
        import re
        json_match = re.search(r'```json\n(.*?)\n```', prompt, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            triangles_data = json.loads(json_str)
            t = triangles_data[0]

            has_vertices = 'vertices' in t
            vertices_count = len(t.get('vertices', []))

            print(f"\n✅ 向后兼容测试成功")
            print(f"   自动构造vertices: {has_vertices}")
            print(f"   顶点数量: {vertices_count}")

            if has_vertices and vertices_count > 0:
                print(f"   构造的vertices:")
                for i, v in enumerate(t['vertices']):
                    print(f"      顶点{i+1}: KP={v['kp']}, 时间={v['time_str']} ({v['time_min']}分钟)")

            return has_vertices and vertices_count >= 3
        else:
            print("❌ 无法从prompt中提取JSON数据")
            return False

    except Exception as e:
        print(f"❌ 向后兼容测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n🧪 开始测试新的混合格式实现")
    print("=" * 80)

    tests = [
        ("混合格式Prompt构建", test_build_prompt_with_sample_data),
        ("数据结构转换", test_triangles_data_structure),
        ("向后兼容性", test_backward_compatibility)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
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
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    success_count = sum(1 for _, result in results if result)
    total_count = len(results)

    print(f"\n总计: {success_count}/{total_count} 测试通过")

    if success_count == total_count:
        print("\n🎉 所有测试通过！新格式实现正确。")
        return True
    else:
        print(f"\n⚠️ 有 {total_count - success_count} 个测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
