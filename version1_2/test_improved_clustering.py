#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进后的聚类功能
"""

from implementation import compiled_agent

def test_improved_clustering():
    """测试改进后的聚类和hull生成"""

    print("=" * 80)
    print("🧪 Testing Improved Clustering and Hull Generation")
    print("=" * 80)

    # 测试用例：関越道 2014年数据
    test_query = "請分析関越道2014年的交通情况"

    print(f"\n📝 Test Query: {test_query}")
    print("-" * 80)

    try:
        # 运行工作流
        result = compiled_agent.invoke({"user_input": test_query})

        print("\n" + "=" * 80)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)

        # 输出结果摘要
        direction_data = result.get("direction_data", {})

        if direction_data:
            print(f"\n📊 Results Summary:")
            print(f"   Total Directions: {len(direction_data)}")

            for direction, data in direction_data.items():
                triangles = data.get("triangles", [])
                clusters = data.get("clusters", [])
                hulls = data.get("hulls", [])

                print(f"\n   Direction: {direction}")
                print(f"      Triangles: {len(triangles)}")
                print(f"      Clusters: {len(clusters)}")
                print(f"      Hulls: {len(hulls)}")

                # 显示Hull图片路径
                hull_fig_path = data.get("hull_fig_path")
                if hull_fig_path:
                    print(f"      Hull Image: {hull_fig_path}")

        # 检查是否所有三角形都被覆盖
        print(f"\n🔍 Coverage Check:")
        for direction, data in direction_data.items():
            triangles = data.get("triangles", [])
            clusters = data.get("clusters", [])

            total_triangles = len(triangles)
            clustered_indices = set(idx for cluster in clusters for idx in cluster)
            covered_count = len(clustered_indices)
            coverage_rate = (covered_count / total_triangles * 100) if total_triangles > 0 else 0

            print(f"   {direction}: {covered_count}/{total_triangles} triangles covered ({coverage_rate:.1f}%)")

            if coverage_rate < 100:
                print(f"      ⚠️  Warning: Not all triangles are covered!")
            else:
                print(f"      ✅ All triangles covered!")

        return result

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_improved_clustering()
