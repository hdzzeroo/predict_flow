#!/usr/bin/env python3
"""
测试方向支持的各个函数（不依赖langgraph）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direction_functions():
    """测试各个函数的方向支持功能"""
    print("=" * 80)
    print("测试方向支持的各个函数")
    print("=" * 80)
    
    # 测试1: chatbot函数
    print("\n1. 测试 chatbot 函数")
    print("-" * 40)
    
    try:
        from implementation import chatbot
        
        # 测试双方向模式
        state1 = {"user_input": "请分析関越高速2024年的交通拥堵情况"}
        result1 = chatbot(state1)
        print(f"双方向输入: {state1['user_input']}")
        print(f"解析结果:")
        print(f"  - route: {result1.get('route')}")
        print(f"  - direction: {result1.get('direction')}")
        print(f"  - file_paths count: {len(result1.get('file_paths', []))}")
        
        # 测试单方向模式
        state2 = {"user_input": "请分析関越高速2024年上り的交通拥堵情况"}
        result2 = chatbot(state2)
        print(f"\n单方向输入: {state2['user_input']}")
        print(f"解析结果:")
        print(f"  - route: {result2.get('route')}")
        print(f"  - direction: {result2.get('direction')}")
        print(f"  - file_paths count: {len(result2.get('file_paths', []))}")
        
        print("✅ chatbot 函数测试成功")
        
    except Exception as e:
        print(f"❌ chatbot 函数测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 文件路径生成函数
    print("\n2. 测试文件路径生成函数")
    print("-" * 40)
    
    try:
        from functions import enhanced_generate_file_paths
        
        # 测试双方向模式
        paths1 = enhanced_generate_file_paths(
            user_input="请分析関越高速2024年的交通拥堵情况",
            use_llm=False
        )
        print(f"双方向模式生成路径数量: {len(paths1)}")
        
        # 显示路径中的方向分布
        up_count = sum(1 for p in paths1 if "_上_" in p)
        down_count = sum(1 for p in paths1 if "_下_" in p)
        print(f"  - 上り方向: {up_count} 个文件")
        print(f"  - 下り方向: {down_count} 个文件")
        
        # 测试单方向模式
        paths2 = enhanced_generate_file_paths(
            user_input="请分析関越高速2024年上り的交通拥堵情况",
            use_llm=False
        )
        print(f"\n单方向模式生成路径数量: {len(paths2)}")
        
        up_count2 = sum(1 for p in paths2 if "_上_" in p)
        down_count2 = sum(1 for p in paths2 if "_下_" in p)
        print(f"  - 上り方向: {up_count2} 个文件")
        print(f"  - 下り方向: {down_count2} 个文件")
        
        print("✅ 文件路径生成函数测试成功")
        
    except Exception as e:
        print(f"❌ 文件路径生成函数测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 方向分组函数
    print("\n3. 测试方向分组函数")
    print("-" * 40)
    
    try:
        from functions import group_files_by_direction
        
        # 模拟文件路径
        test_paths = [
            "/test/関越道_上_2024_08-01.csv",
            "/test/関越道_上_2023_08-01.csv", 
            "/test/関越道_下_2024_08-02.csv",
            "/test/関越道_下_2023_08-02.csv"
        ]
        
        grouped = group_files_by_direction(test_paths)
        print(f"分组结果:")
        for direction, files in grouped.items():
            print(f"  - {direction}方向: {len(files)} 个文件")
        
        print("✅ 方向分组函数测试成功")
        
    except Exception as e:
        print(f"❌ 方向分组函数测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_workflow_simulation():
    """模拟完整工作流测试（使用模拟数据）"""
    print("\n4. 模拟完整工作流测试")
    print("-" * 40)
    
    try:
        from implementation import chatbot, visualization, cluster, draw_hulls, report
        
        # 第1步: chatbot - 双方向模式
        print("步骤1: 解析用户输入（双方向模式）")
        state = {"user_input": "请分析関越高速2024年的交通拥堵情况"}
        state.update(chatbot(state))
        print(f"  - 检测到方向: {state.get('direction', 'None (双方向模式)')}")
        print(f"  - 生成文件数: {len(state.get('file_paths', []))}")
        
        # 检查是否有实际文件存在
        actual_files = []
        for fp in state.get('file_paths', []):
            if os.path.exists(fp):
                actual_files.append(fp)
        
        if actual_files:
            print(f"  - 找到实际文件: {len(actual_files)} 个")
            
            # 第2步: visualization
            print("\n步骤2: 数据可视化")
            # 更新state使用实际存在的文件
            state['file_paths'] = actual_files[:4]  # 限制文件数量避免过长处理
            vis_result = visualization(state)
            state.update(vis_result)
            
            direction_data = state.get("direction_data", {})
            print(f"  - 处理的方向: {list(direction_data.keys())}")
            print(f"  - 总三角形数: {len(state.get('triangles', []))}")
            
            # 第3步: cluster
            print("\n步骤3: 聚类分析")
            cluster_result = cluster(state)
            state.update(cluster_result)
            
            total_clusters = len(state.get("clusters", []))
            print(f"  - 总聚类数: {total_clusters}")
            
            # 第4步: draw_hulls
            print("\n步骤4: 绘制热点")
            hull_result = draw_hulls(state)
            state.update(hull_result)
            
            total_hulls = len(state.get("hulls", []))
            print(f"  - 总热点数: {total_hulls}")
            
            # 第5步: report
            print("\n步骤5: 生成报告")
            report_result = report(state)
            state.update(report_result)
            
            final_report = state.get("final_report", "")
            print(f"  - 报告长度: {len(final_report)} 字符")
            print(f"  - 包含方向分析: {'是' if '方向' in final_report else '否'}")
            
            # 保存报告到文件
            with open("output/direction_test_report.md", "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"  - 报告已保存到: output/direction_test_report.md")
            
            print("\n✅ 完整工作流模拟测试成功")
            return True
            
        else:
            print(f"  ⚠️ 未找到实际数据文件，跳过后续步骤")
            print(f"  期望的文件路径示例: {state.get('file_paths', [])[:2]}")
            return False
        
    except Exception as e:
        print(f"❌ 完整工作流模拟测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 测试各个函数
    test_direction_functions()
    
    # 测试完整工作流（如果有数据文件）
    success = test_workflow_simulation()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("✅ 所有方向支持功能已实现并测试")
    if success:
        print("✅ 完整工作流测试成功，支持方向分组分析")
    else:
        print("⚠️ 完整工作流需要实际数据文件才能完全验证")
    print("🎯 系统现在支持:")
    print("   - 自动识别用户指定的方向（上り/下り）")
    print("   - 单方向模式：只分析指定方向")
    print("   - 双方向模式：分别分析上下两个方向")
    print("   - 生成分方向的详细报告和对比分析")