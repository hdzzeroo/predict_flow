#!/usr/bin/env python3
"""
测试完整的方向支持工作流
"""

from implementation import compiled_agent
import json
import os

def test_direction_workflow():
    """测试方向支持的完整工作流"""
    print("=" * 80)
    print("测试方向支持的完整工作流")
    print("=" * 80)
    
    # 测试案例
    test_cases = [
        {
            "name": "单方向模式 - 上り方向",
            "input": "请分析関越高速2024年上り的交通拥堵情况",
            "expected_mode": "single_direction"
        },
        {
            "name": "单方向模式 - 下り方向", 
            "input": "请分析関越高速2024年下り的交通拥堵情况",
            "expected_mode": "single_direction"
        },
        {
            "name": "双方向模式 - 未指定方向",
            "input": "请分析関越高速2024年的交通拥堵情况",
            "expected_mode": "dual_direction"
        }
    ]
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试 {test_case['name']}")
        print("-" * 60)
        print(f"输入: {test_case['input']}")
        
        try:
            # 执行完整工作流
            result = compiled_agent.invoke({"user_input": test_case['input']})
            
            # 分析结果
            print(f"\n✅ 工作流执行成功")
            
            # 检查基本字段
            basic_fields = ["route", "direction", "triangles", "clusters", "hulls", "final_report"]
            for field in basic_fields:
                if field in result:
                    print(f"  ✓ {field}: 存在")
                else:
                    print(f"  ✗ {field}: 缺失")
            
            # 检查方向数据
            direction_data = result.get("direction_data", {})
            if direction_data:
                print(f"  ✓ direction_data: 包含 {list(direction_data.keys())} 方向")
                
                # 验证每个方向的数据完整性
                for direction, data in direction_data.items():
                    required_keys = ["triangles", "clusters", "hulls", "fig_path"]
                    missing_keys = [key for key in required_keys if key not in data]
                    if missing_keys:
                        print(f"  ⚠️ {direction}方向缺失字段: {missing_keys}")
                    else:
                        print(f"  ✓ {direction}方向: 数据完整 ({len(data['triangles'])} triangles, {len(data['clusters'])} clusters, {len(data['hulls'])} hulls)")
            else:
                print(f"  ⚠️ direction_data: 未找到方向分组数据")
            
            # 验证模式
            user_direction = result.get("direction")
            if test_case['expected_mode'] == "single_direction":
                if user_direction:
                    print(f"  ✓ 模式验证: 单方向模式 ({user_direction})")
                    # 应该只有一个方向的数据
                    if direction_data and len(direction_data) == 1:
                        print(f"  ✓ 方向数据: 符合单方向模式")
                    else:
                        print(f"  ⚠️ 方向数据: 不符合单方向模式 (期望1个方向，实际{len(direction_data)}个)")
                else:
                    print(f"  ⚠️ 模式验证: 单方向模式失败，未检测到指定方向")
            
            elif test_case['expected_mode'] == "dual_direction":
                if not user_direction:
                    print(f"  ✓ 模式验证: 双方向模式")
                    # 应该有两个方向的数据
                    if direction_data and len(direction_data) == 2:
                        print(f"  ✓ 方向数据: 符合双方向模式")
                    else:
                        print(f"  ⚠️ 方向数据: 不符合双方向模式 (期望2个方向，实际{len(direction_data)}个)")
                else:
                    print(f"  ⚠️ 模式验证: 双方向模式失败，意外检测到方向 {user_direction}")
            
            # 检查报告内容
            final_report = result.get("final_report", "")
            if final_report:
                report_lines = len(final_report.split('\n'))
                print(f"  ✓ 报告生成: {len(final_report)} 字符, {report_lines} 行")
                
                # 检查报告是否包含方向信息
                if "上り方向" in final_report or "下り方向" in final_report:
                    print(f"  ✓ 报告内容: 包含方向分析")
                else:
                    print(f"  ⚠️ 报告内容: 未包含方向分析")
            else:
                print(f"  ✗ 报告生成: 失败")
            
            # 保存测试结果
            output_file = f"output/test_result_{i}_{test_case['expected_mode']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                # 过滤掉过大的字段以便保存
                filtered_result = {
                    "test_case": test_case,
                    "route": result.get("route"),
                    "direction": result.get("direction"),
                    "triangle_count": len(result.get("triangles", [])),
                    "cluster_count": len(result.get("clusters", [])),
                    "hull_count": len(result.get("hulls", [])),
                    "direction_data_summary": {
                        direction: {
                            "triangles": len(data.get("triangles", [])),
                            "clusters": len(data.get("clusters", [])),
                            "hulls": len(data.get("hulls", [])),
                            "has_fig_path": bool(data.get("fig_path"))
                        }
                        for direction, data in direction_data.items()
                    } if direction_data else {},
                    "report_length": len(final_report),
                    "success": True
                }
                json.dump(filtered_result, f, ensure_ascii=False, indent=2)
            
            print(f"  📁 测试结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 工作流执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 保存错误信息
            error_file = f"output/test_error_{i}_{test_case['expected_mode']}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

def quick_demo():
    """快速演示：只测试双方向模式"""
    print("=" * 60)
    print("快速演示：双方向模式")
    print("=" * 60)
    
    test_input = "请分析関越高速2024年的交通拥堵情况"
    print(f"输入: {test_input}")
    
    try:
        result = compiled_agent.invoke({"user_input": test_input})
        
        direction_data = result.get("direction_data", {})
        print(f"\n处理的方向: {list(direction_data.keys())}")
        
        for direction, data in direction_data.items():
            print(f"\n{direction}方向:")
            print(f"  - 拥堵事件: {len(data.get('triangles', []))}")
            print(f"  - 聚类: {len(data.get('clusters', []))}")
            print(f"  - 热点: {len(data.get('hulls', []))}")
        
        print(f"\n报告长度: {len(result.get('final_report', ''))} 字符")
        print("✅ 演示完成")
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        test_direction_workflow()