"""
测试聚类改进效果
验证新的严格聚类规则是否解决了过度合并问题
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from implementation import compiled_agent
from config import config

def validate_cluster(cluster_triangles, triangles_data, cluster_id):
    """
    验证聚类质量

    Returns:
        (is_valid, issues_list)
    """
    issues = []

    if len(cluster_triangles) < 2:
        return True, []

    # 提取聚类中所有三角形的中心点
    centers = []
    for tri_id in cluster_triangles:
        if tri_id < len(triangles_data):
            t = triangles_data[tri_id]
            center = t.get('center', [0, 0])
            centers.append((tri_id, center))

    # 检查1：两两距离
    max_spatial = 0
    max_temporal = 0
    violations = []

    for i, (id1, c1) in enumerate(centers):
        for id2, c2 in centers[i+1:]:
            spatial_dist = abs(c1[0] - c2[0])
            temporal_dist = abs(c1[1] - c2[1])

            max_spatial = max(max_spatial, spatial_dist)
            max_temporal = max(max_temporal, temporal_dist)

            if spatial_dist > 7.0 or temporal_dist > 180:
                violations.append({
                    'pair': f'{id1}-{id2}',
                    'spatial': round(spatial_dist, 1),
                    'temporal': round(temporal_dist, 0)
                })

    if violations:
        issues.append(f"❌ 两两距离违反: {len(violations)}对不满足阈值")
        for v in violations[:3]:  # 只显示前3个
            issues.append(f"   {v['pair']}: 空间{v['spatial']}km, 时间{v['temporal']}min")
        if len(violations) > 3:
            issues.append(f"   ...还有{len(violations)-3}对违反")

    # 检查2：KP跨度
    kp_values = [c[0] for _, c in centers]
    kp_span = max(kp_values) - min(kp_values)

    if kp_span > 35.0:
        issues.append(f"❌ KP跨度过大: {kp_span:.1f}km > 35km")

    # 检查3：时间跨度
    time_values = [c[1] for _, c in centers]
    time_span = max(time_values) - min(time_values)

    if time_span > 360:
        issues.append(f"❌ 时间跨度过大: {time_span:.0f}分钟 > 360分钟")

    is_valid = len(issues) == 0

    if is_valid:
        summary = (f"✅ Cluster {cluster_id}: "
                  f"KP跨度{kp_span:.1f}km, "
                  f"时间跨度{time_span:.0f}min, "
                  f"最大两两距离: 空间{max_spatial:.1f}km, 时间{max_temporal:.0f}min")
        return True, [summary]
    else:
        return False, issues


def test_clustering_quality():
    """测试聚类质量"""

    print("="*70)
    print("🧪 测试聚类改进效果")
    print("="*70)

    # 检查API密钥
    if not config.is_llm_available():
        print("❌ 错误: 需要设置 OPENAI_API_KEY 环境变量")
        return

    print(f"\n✅ 使用模型: {config.openai_model}")

    # 测试输入
    test_input = "请分析2025年5月4日関越道的交通拥堵情况"

    print(f"\n📝 测试输入: {test_input}")
    print("\n开始分析...")
    print("-"*70)

    try:
        # 执行分析
        result = compiled_agent.invoke({
            "user_input": test_input
        })

        # 获取数据
        llm_analysis = result.get('llm_analysis', {})
        direction_data = result.get('direction_data', {})

        if not llm_analysis:
            print("❌ 错误: 没有LLM分析结果")
            return

        # 检查每个方向
        for direction in ['上', '下']:
            if direction not in llm_analysis:
                continue

            print(f"\n{'='*70}")
            print(f"📊 {direction}り方向聚类质量检查")
            print(f"{'='*70}")

            analysis = llm_analysis[direction]
            hotspots = analysis.get('hotspots', [])
            triangles = direction_data.get(direction, {}).get('triangles', [])

            print(f"\n总热点数: {len(hotspots)}")
            print(f"总三角形数: {len(triangles)}")

            # 验证每个热点
            all_valid = True
            valid_count = 0
            invalid_count = 0

            for hotspot in hotspots:
                hotspot_id = hotspot['hotspot_id']
                cluster_ids = hotspot.get('included_triangle_ids', [])

                print(f"\n--- Hotspot {hotspot_id} (包含{len(cluster_ids)}个事件) ---")

                is_valid, messages = validate_cluster(cluster_ids, triangles, hotspot_id)

                for msg in messages:
                    print(msg)

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    all_valid = False

            # 总结
            print(f"\n{'='*70}")
            print(f"📈 {direction}り方向聚类质量总结")
            print(f"{'='*70}")
            print(f"✅ 合格聚类: {valid_count}/{len(hotspots)}")
            print(f"❌ 不合格聚类: {invalid_count}/{len(hotspots)}")

            if all_valid:
                print(f"\n🎉 {direction}り方向所有聚类质量合格！")
            else:
                print(f"\n⚠️ {direction}り方向存在{invalid_count}个不合格聚类，需要改进")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_clustering_quality()
