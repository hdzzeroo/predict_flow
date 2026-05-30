#!/usr/bin/env python3
"""
简单的工作流测试脚本（无依赖版本）
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from excel_output_generator import ExcelOutputGenerator, generate_csv_prediction_output
        print("✅ excel_output_generator 导入成功")
        
        # 测试基本功能
        generator = ExcelOutputGenerator()
        print("✅ ExcelOutputGenerator 实例化成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_generation():
    """测试CSV生成功能"""
    print("\n📄 测试CSV生成功能...")
    
    try:
        from excel_output_generator import ExcelOutputGenerator
        
        # 简单的模拟数据
        mock_data = {
            "route": "関越道",
            "ts": "2024年",
            "direction_data": {
                "上": {
                    "hulls": [
                        {
                            "cluster_id": 0,
                            "cluster_size": 3,
                            "kp_center": 45.0,
                            "time_center": 600.0,  # 10:00
                            "kp_range": [40.0, 50.0],
                            "time_range": [540, 660]
                        }
                    ]
                }
            }
        }
        
        generator = ExcelOutputGenerator()
        df = generator.convert_workflow_to_csv_format(mock_data)
        
        if df.empty:
            print("❌ 生成的DataFrame为空")
            return False
            
        print(f"✅ CSV数据生成成功: {len(df)}行 x {len(df.columns)}列")
        
        # 检查关键字段
        if len(df) > 0:
            row = df.iloc[0]
            print(f"   - 道路コード: {row[0]}")
            print(f"   - 道路名: {row[1]}")
            print(f"   - 方向: {row[3]}")
            print(f"   - KP: {row[7]}")
            print(f"   - 時間帯: {row[8]}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_save():
    """测试文件保存功能"""
    print("\n💾 测试文件保存功能...")
    
    try:
        from excel_output_generator import generate_csv_prediction_output
        
        # 模拟数据
        mock_data = {
            "route": "関越道",
            "direction_data": {
                "下": {
                    "hulls": [
                        {
                            "cluster_id": 0,
                            "cluster_size": 2,
                            "kp_center": 30.0,
                            "time_center": 1080.0  # 18:00
                        }
                    ]
                }
            }
        }
        
        output_path = generate_csv_prediction_output(mock_data)
        print(f"✅ 文件保存成功: {output_path}")
        
        # 验证文件存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ 文件验证成功: 大小 {file_size} bytes")
            
            # 读取前几行检查
            with open(output_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()[:5]
                print(f"✅ 文件内容预览: 共{len(lines)}行")
                for i, line in enumerate(lines):
                    print(f"   行{i+1}: {line.strip()[:80]}...")
        else:
            print(f"❌ 文件不存在: {output_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 文件保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_test():
    """运行简单测试"""
    
    print("=" * 50)
    print("简单工作流测试")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("CSV生成测试", test_csv_generation),
        ("文件保存测试", test_file_save)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 运行 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("\n🎉 所有基础测试通过！")
        print("CSV输出功能基础功能正常。")
        print("\n下一步可以:")
        print("1. 运行完整的工作流测试")
        print("2. 检查生成的CSV文件格式")
        print("3. 对比模板文件确保格式正确")
    else:
        print(f"\n⚠️ 有 {len(results)-passed} 个测试失败")
        print("请检查错误信息并修复问题。")

if __name__ == "__main__":
    run_simple_test()