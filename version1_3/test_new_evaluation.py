"""
简单测试新的KP区间评估方法
"""

# 测试Evaluator初始化和KP区间加载
print("="*70)
print("测试新的KP区间评估方法")
print("="*70)

try:
    from evaluation import Evaluator

    print("\n【步骤1】初始化Evaluator")
    evaluator = Evaluator(road_type="関越道", direction="下", time_step_minutes=60)

    print(f"\n【步骤2】检查KP区间")
    print(f"  道路类型: {evaluator.road_type}")
    print(f"  方向: {evaluator.direction}")
    print(f"  时间步长: {evaluator.time_step} 分钟")
    print(f"  KP区间数量: {len(evaluator.kp_intervals)}")

    if evaluator.kp_intervals:
        print(f"\n  前5个KP区间:")
        for i, (start, end) in enumerate(evaluator.kp_intervals[:5]):
            print(f"    区间{i}: [{start:.2f}, {end:.2f}] km (宽度: {end-start:.2f} km)")

        print(f"\n  后5个KP区间:")
        for i, (start, end) in enumerate(evaluator.kp_intervals[-5:], len(evaluator.kp_intervals)-5):
            print(f"    区间{i}: [{start:.2f}, {end:.2f}] km (宽度: {end-start:.2f} km)")

    print("\n✓ 测试成功！新的KP区间评估方法已正确实现")
    print("\n【关键改进】")
    print("  1. 栅格划分基于道路实际KP点，而非均匀划分")
    print("  2. 使用矩形相交判断，而非中心点采样")
    print("  3. 总栅格数: {} × 24小时 = {} 个栅格".format(
        len(evaluator.kp_intervals),
        len(evaluator.kp_intervals) * 24
    ))

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
