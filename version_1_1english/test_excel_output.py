#!/usr/bin/env python3
"""
测试Excel输出功能
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation import compiled_agent
from excel_output_generator import ExcelOutputGenerator
import pandas as pd

def test_excel_output():
    """测试完整的工作流和Excel输出"""
    
    print("🧪 开始测试Excel输出功能...")
    
    # 测试用例
    test_cases = [
        {
            "input": "请分析関越高速2024年上り的交通拥堵情况",
            "description": "単方向模式测试 - 上り"
        },
        {
            "input": "请分析関越高速2024年下り的交通拥堵情况", 
            "description": "単方向模式测试 - 下り"
        },
        {
            "input": "请分析関越高速2024年的交通拥堵情况",
            "description": "双方向模式测试"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试 {i}: {test_case['description']}")
        print(f"输入: {test_case['input']}")
        print('='*50)
        
        try:
            # 运行完整工作流
            result = compiled_agent.invoke({"user_input": test_case["input"]})
            
            # 检查结果
            if "final_report" in result:
                print(f"✅ 文本报告生成成功 (长度: {len(result['final_report'])} 字符)")
            else:
                print("❌ 文本报告生成失败")
                continue
                
            if "excel_output_path" in result and result["excel_output_path"]:
                excel_path = result["excel_output_path"]
                print(f"✅ Excel输出生成成功: {excel_path}")
                
                # 验证Excel文件
                if os.path.exists(excel_path):
                    try:
                        df = pd.read_excel(excel_path)
                        print(f"✅ Excel文件验证成功:")
                        print(f"   - 行数: {len(df)}")
                        print(f"   - 列数: {len(df.columns)}")
                        if len(df) > 0:
                            print(f"   - 列名: {list(df.columns)}")
                            print(f"   - 示例数据:")
                            print(df.head(2).to_string(index=False))
                    except Exception as e:
                        print(f"⚠️ Excel文件读取失败: {e}")
                else:
                    print(f"❌ Excel文件不存在: {excel_path}")
                    continue
                    
                success_count += 1
                    
            else:
                print("❌ Excel输出路径为空或未生成")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n{'='*50}")
    print(f"测试总结: {success_count}/{len(test_cases)} 个测试用例成功")
    print('='*50)
    
    return success_count == len(test_cases)

def test_excel_generator_directly():
    """直接测试Excel生成器"""
    
    print("\n🔬 直接测试Excel生成器...")
    
    # 创建模拟的工作流结果
    mock_result = {
        "route": "関越道",
        "ts": "2024年",
        "direction_data": {
            "上": {
                "triangles": [
                    {"id": 0, "center": [50.0, 540.0], "area": 100.0}
                ],
                "hulls": [
                    {
                        "cluster_id": 0,
                        "cluster_size": 5,
                        "kp_center": 50.0,
                        "time_center": 540.0,  # 9:00
                        "kp_range": [45.0, 55.0],
                        "time_range": [480, 600],  # 8:00-10:00
                        "width": 10.0,
                        "height": 120.0,
                        "area": 1200.0
                    }
                ]
            }
        }
    }
    
    try:
        generator = ExcelOutputGenerator()
        df = generator.convert_workflow_to_excel_format(mock_result)
        
        print("✅ 直接测试成功:")
        print(f"   - 生成行数: {len(df)}")
        if len(df) > 0:
            print(f"   - 列名: {list(df.columns)}")
            print("   - 示例数据:")
            print(df.to_string(index=False))
            
        # 保存到文件
        output_path = "/home/dizhihuang/graduate/predict_workflow/version_1_1english/output/test_direct_excel_output.xlsx"
        saved_path = generator.save_excel_output(mock_result, output_path)
        print(f"✅ 文件保存成功: {saved_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始Excel输出功能测试...")
    
    # 先测试生成器本身
    direct_test_ok = test_excel_generator_directly()
    
    if direct_test_ok:
        # 再测试完整工作流
        full_test_ok = test_excel_output()
        
        if full_test_ok:
            print("\n🎉 所有测试通过！Excel输出功能已成功集成。")
        else:
            print("\n⚠️ 完整工作流测试未完全通过，请检查集成。")
    else:
        print("\n❌ Excel生成器基础测试失败，请检查代码。")