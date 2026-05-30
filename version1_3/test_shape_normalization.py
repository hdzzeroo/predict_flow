"""
测试形状规范化功能
验证三角形和四边形的顶点清理是否正常工作
"""

from data_preparers import DataPreparer

def test_shape_normalization():
    """测试形状规范化功能"""

    print("="*70)
    print("测试形状规范化功能")
    print("="*70)

    # 测试数据：包含不同类型的三角形和四边形
    test_triangles = [
        {
            "id": 0,
            "shape_type": "triangle",
            "vertices": [
                [0.0, 980],
                [3.1, 1050],
                [0.0, 1135]
            ],
            "kp_start": 0.0,
            "kp_end": 3.1
        },
        {
            "id": 1,
            "shape_type": "quadrilateral",
            "vertices": [
                [31.4, 1210],
                [64.4, 1210],  # 重复的顶点
                [64.4, 1210],  # 重复的顶点
                [31.4, 1320]
            ],
            "kp_start": 31.4,
            "kp_end": 64.4
        },
        {
            "id": 2,
            "shape_type": "triangle",
            "vertices": [
                [33.0, 925],
                [45.5, 965],
                [33.0, 975]
            ],
            "kp_start": 33.0,
            "kp_end": 45.5
        },
        {
            "id": 3,
            "shape_type": "quadrilateral",
            "vertices": [
                [33.0, 1080],
                [49.5, 1080],
                [49.5, 1080],  # 重复的顶点
                [33.0, 1095]
            ],
            "kp_start": 33.0,
            "kp_end": 49.5
        },
        {
            "id": 4,
            "shape_type": "quadrilateral",
            "vertices": [
                [10.0, 500],
                [20.0, 500],
                [20.0, 600],
                [10.0, 600]
            ],
            "kp_start": 10.0,
            "kp_end": 20.0
        }
    ]

    print(f"\n原始数据：{len(test_triangles)} 个形状")
    print("-" * 70)
    for t in test_triangles:
        print(f"ID {t['id']}: {t['shape_type']} - {len(t['vertices'])} 个顶点")
        for i, v in enumerate(t['vertices']):
            print(f"  顶点{i+1}: KP={v[0]:.1f}, Time={v[1]}")

    # 应用规范化
    normalized = DataPreparer.normalize_triangle_shapes(test_triangles)

    print(f"\n\n清理后数据：{len(normalized)} 个形状")
    print("-" * 70)
    for t in normalized:
        print(f"ID {t['id']}: {t['shape_type']} - {len(t['vertices'])} 个顶点")
        for i, v in enumerate(t['vertices']):
            print(f"  顶点{i+1}: KP={v[0]:.1f}, Time={v[1]}")
        print()

    # 验证结果
    print("\n" + "="*70)
    print("验证结果:")
    print("="*70)

    passed = True

    # 检查 ID 0: 应该保持为三角形（3个顶点）
    t0 = next((t for t in normalized if t['id'] == 0), None)
    if t0 and t0['shape_type'] == 'triangle' and len(t0['vertices']) == 3:
        print("✅ ID 0: 保持为三角形 (3个顶点)")
    else:
        print("❌ ID 0: 测试失败")
        passed = False

    # 检查 ID 1: 应该从四边形降级为三角形（去除重复顶点后只有3个）
    t1 = next((t for t in normalized if t['id'] == 1), None)
    if t1 and t1['shape_type'] == 'triangle' and len(t1['vertices']) == 3:
        print("✅ ID 1: 从四边形降级为三角形 (去除重复顶点)")
    else:
        print("❌ ID 1: 测试失败")
        passed = False

    # 检查 ID 2: 应该保持为三角形（3个顶点）
    t2 = next((t for t in normalized if t['id'] == 2), None)
    if t2 and t2['shape_type'] == 'triangle' and len(t2['vertices']) == 3:
        print("✅ ID 2: 保持为三角形 (3个顶点)")
    else:
        print("❌ ID 2: 测试失败")
        passed = False

    # 检查 ID 3: 应该从四边形降级为三角形
    t3 = next((t for t in normalized if t['id'] == 3), None)
    if t3 and t3['shape_type'] == 'triangle' and len(t3['vertices']) == 3:
        print("✅ ID 3: 从四边形降级为三角形 (去除重复顶点)")
    else:
        print("❌ ID 3: 测试失败")
        passed = False

    # 检查 ID 4: 应该保持为四边形（4个不同的顶点）
    t4 = next((t for t in normalized if t['id'] == 4), None)
    if t4 and t4['shape_type'] == 'quadrilateral' and len(t4['vertices']) == 4:
        print("✅ ID 4: 保持为四边形 (4个不同的顶点)")
    else:
        print("❌ ID 4: 测试失败")
        passed = False

    print("\n" + "="*70)
    if passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*70)

    return passed


if __name__ == "__main__":
    test_shape_normalization()
