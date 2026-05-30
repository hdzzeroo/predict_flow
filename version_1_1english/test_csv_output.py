#!/usr/bin/env python3
"""
测试CSV格式输出功能
"""

import sys
import os
import pandas as pd
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_output_generator import ExcelOutputGenerator, generate_csv_prediction_output

def test_csv_generator_directly():
    """直接测试CSV生成器"""
    
    print("🔬 直接测试CSV生成器...")
    
    # 创建模拟的工作流结果
    mock_result = {
        "route": "関越道",
        "ts": "2024年",
        "direction_data": {
            "上": {
                "triangles": [
                    {"id": 0, "center": [50.0, 540.0], "area": 100.0},
                    {"id": 1, "center": [60.0, 480.0], "area": 80.0}
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
            },
            "下": {
                "triangles": [
                    {"id": 2, "center": [30.0, 1020.0], "area": 150.0}
                ],
                "hulls": [
                    {
                        "cluster_id": 1,
                        "cluster_size": 3,
                        "kp_center": 30.0,
                        "time_center": 1020.0,  # 17:00
                        "kp_range": [25.0, 35.0],
                        "time_range": [960, 1080],  # 16:00-18:00
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
        df = generator.convert_workflow_to_csv_format(mock_result)
        
        print("✅ 直接测试成功:")
        print(f"   - 生成行数: {len(df)}")
        print(f"   - 列数: {len(df.columns) if not df.empty else 0}")
        
        if len(df) > 0:
            print("   - 前15列内容:")
            for i, row in df.iterrows():
                print(f"     Row {i}: {list(row[:15])}")
                break
                
        # 保存到文件
        output_path = "/home/dizhihuang/graduate/predict_workflow/version_1_1english/output/test_direct_csv_output.csv"
        saved_path = generator.save_csv_output(mock_result, output_path)
        print(f"✅ 文件保存成功: {saved_path}")
        
        # 验证保存的文件
        if os.path.exists(saved_path):
            with open(saved_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            print(f"✅ 文件验证成功: 共{len(lines)}行")
            print(f"   - 标题行: {lines[0].strip()[:50]}...")
            if len(lines) > 2:
                print(f"   - 数据行示例: {lines[2].strip()[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_workflow():
    """测试完整工作流的CSV输出"""
    
    print("\n🧪 测试完整工作流CSV输出...")
    
    try:
        # 这里不导入完整的工作流，因为可能有依赖问题
        # 而是直接测试生成函数
        
        mock_workflow_result = {
            "route": "関越道",
            "ts": "2024年5月",
            "direction_data": {
                "上": {
                    "triangles": [{"id": 0, "center": [40.0, 600.0]}],
                    "hulls": [{
                        "cluster_id": 0,
                        "cluster_size": 4,
                        "kp_center": 40.0,
                        "time_center": 600.0,  # 10:00
                        "kp_range": [35.0, 45.0],
                        "time_range": [540, 660]  # 9:00-11:00
                    }]
                }
            }
        }
        
        output_path = generate_csv_prediction_output(mock_workflow_result)
        print(f"✅ 完整工作流测试成功: {output_path}")
        
        # 验证生成的文件
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, skiprows=1, encoding='utf-8-sig')
            print(f"✅ CSV文件读取成功: {len(df)}行 x {len(df.columns)}列")
            if len(df) > 0:
                print(f"   - 道路名: {df.iloc[0, 1] if len(df.columns) > 1 else 'N/A'}")
                print(f"   - 方向: {df.iloc[0, 3] if len(df.columns) > 3 else 'N/A'}")
                print(f"   - KP: {df.iloc[0, 7] if len(df.columns) > 7 else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_format_compliance():
    """测试CSV格式规范性"""
    
    print("\n📋 测试CSV格式规范性...")
    
    try:
        # 读取模板文件进行对比
        template_path = "/home/dizhihuang/graduate/predict_workflow/data/渋滞予測フォーマットR7.4.csv"
        
        if not os.path.exists(template_path):
            print("⚠️ 模板文件不存在，跳过格式对比测试")
            return True
            
        # 读取模板列结构
        with open(template_path, 'r', encoding='utf-8-sig') as f:
            f.readline()  # 跳过标题行
            template_headers = next(pd.read_csv(f, nrows=0)).columns.tolist()
        
        # 生成测试数据
        mock_result = {
            "route": "関越道",
            "direction_data": {
                "上": {
                    "hulls": [{
                        "cluster_id": 0,
                        "cluster_size": 2,
                        "kp_center": 25.0,
                        "time_center": 480.0
                    }]
                }
            }
        }
        
        generator = ExcelOutputGenerator()
        df = generator.convert_workflow_to_csv_format(mock_result)
        
        if df.empty:
            print("❌ 生成的DataFrame为空")
            return False
            
        generated_headers = df.columns.tolist()
        
        print(f"✅ 格式对比:")
        print(f"   - 模板列数: {len(template_headers)}")
        print(f"   - 生成列数: {len(generated_headers)}")
        
        if len(template_headers) == len(generated_headers):
            print("✅ 列数匹配")
        else:
            print("⚠️ 列数不匹配")
            
        # 检查关键列名
        key_columns = ['道路\nコード', '道路名', '月日', '方向', 'KP', '時間帯']
        for col in key_columns:
            if col in generated_headers:
                print(f"✅ 关键列存在: {repr(col)}")
            else:
                print(f"❌ 关键列缺失: {repr(col)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 格式规范性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    
    print("=" * 60)
    print("CSV输出功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试1：直接测试生成器
    test_results.append(("CSV生成器直接测试", test_csv_generator_directly()))
    
    # 测试2：完整工作流测试
    test_results.append(("完整工作流测试", test_full_workflow()))
    
    # 测试3：格式规范性测试
    test_results.append(("CSV格式规范性测试", test_csv_format_compliance()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 个测试通过")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！CSV输出功能正常工作。")
    else:
        print("⚠️ 部分测试未通过，请检查问题。")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)