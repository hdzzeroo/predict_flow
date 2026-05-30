#!/usr/bin/env python3
"""
测试改进后的CSV生成功能
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from excel_output_generator import ExcelOutputGenerator, generate_csv_prediction_output

def create_test_workflow_result():
    """创建测试用的工作流结果数据"""
    return {
        "route": "関越道",
        "ts": "Year 2024, Month 4, Day 30",
        "direction_data": {
            "上": {
                "hulls": [
                    {
                        "cluster_id": 0,
                        "cluster_size": 5,
                        "kp_center": 15.5,
                        "time_center": 480,  # 8:00 (8*60 = 480分钟)
                        "kp_range": [10.0, 20.0],
                        "time_range": [420, 540],  # 7:00-9:00 (420-540分钟)
                        "center": [15.5, 480],
                        "area": 1000.0,
                        "width": 10.0,
                        "height": 120
                    },
                    {
                        "cluster_id": 1,
                        "cluster_size": 3,
                        "kp_center": 35.2,
                        "time_center": 1080,  # 18:00 (18*60 = 1080分钟)
                        "kp_range": [30.0, 40.0],
                        "time_range": [1020, 1140],  # 17:00-19:00
                        "center": [35.2, 1080],
                        "area": 600.0,
                        "width": 10.0,
                        "height": 120
                    }
                ]
            },
            "下": {
                "hulls": [
                    {
                        "cluster_id": 2,
                        "cluster_size": 4,
                        "kp_center": 25.0,
                        "time_center": 600,  # 10:00
                        "kp_range": [20.0, 30.0],
                        "time_range": [540, 660],  # 9:00-11:00
                        "center": [25.0, 600],
                        "area": 800.0,
                        "width": 10.0,
                        "height": 120
                    }
                ]
            }
        }
    }

def test_time_header_generation():
    """测试时间标题行生成"""
    print("🧪 测试时间标题行生成...")
    
    generator = ExcelOutputGenerator()
    time_headers = generator.generate_time_header_row()
    
    print(f"✅ 时间标题行长度: {len(time_headers)} 列")
    print("前20列标题:")
    for i, header in enumerate(time_headers[:20]):
        print(f"  {i+1:2d}. {header}")
    
    print("\n时间列标题 (第16-25列):")
    for i, header in enumerate(time_headers[15:25]):
        print(f"  {i+16:2d}. {header}")
    
    print(f"\n最后3列标题:")
    for i, header in enumerate(time_headers[-3:]):
        print(f"  {len(time_headers)-3+i+1:2d}. {header}")

def test_congestion_pattern_calculation():
    """测试拥堵分布模式计算"""
    print("\n🧪 测试拥堵分布模式计算...")
    
    generator = ExcelOutputGenerator()
    
    # 测试用的hull数据
    test_hull = {
        "cluster_id": 0,
        "cluster_size": 5,
        "time_center": 480,  # 8:00
        "time_range": [420, 540],  # 7:00-9:00
        "center": [15.5, 480]
    }
    
    pattern = generator.calculate_congestion_pattern(test_hull, peak_congestion_km=5.0)
    
    print(f"✅ 拥堵分布模式长度: {len(pattern)} 小时")
    
    # 显示非空的时间段
    non_empty_hours = []
    for hour, value in enumerate(pattern):
        if value and value != '':
            non_empty_hours.append((hour, value))
    
    print("拥堵时间段:")
    for hour, value in non_empty_hours:
        print(f"  {hour:2d}时: {value}km")

def test_csv_generation():
    """测试完整的CSV生成"""
    print("\n🧪 测试完整的CSV生成...")
    
    # 创建测试数据
    workflow_result = create_test_workflow_result()
    
    # 生成CSV文件
    output_path = generate_csv_prediction_output(workflow_result, output_dir="output")
    
    print(f"✅ CSV文件已生成: {output_path}")
    
    # 读取并显示前几行
    if os.path.exists(output_path):
        print("\nCSV文件内容预览:")
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:5]):  # 显示前5行
                print(f"  行{i+1}: {line.strip()[:100]}...")  # 只显示前100个字符
        
        print(f"\nCSV文件总行数: {len(lines)}")
    else:
        print("❌ CSV文件生成失败")

def main():
    """主测试函数"""
    print("🚀 开始测试改进后的CSV生成功能")
    print("=" * 60)
    
    try:
        # 测试时间标题行生成
        test_time_header_generation()
        
        # 测试拥堵分布模式计算
        test_congestion_pattern_calculation()
        
        # 测试完整CSV生成
        test_csv_generation()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()











