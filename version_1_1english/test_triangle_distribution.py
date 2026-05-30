#!/usr/bin/env python3
"""
测试三角形分布的拥堵模式计算
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from excel_output_generator import ExcelOutputGenerator

def test_detailed_triangle_distribution():
    """测试详细的三角形分布"""
    print("🧪 测试三角形拥堵分布模式...")
    
    generator = ExcelOutputGenerator()
    
    # 测试用的hull数据 - 更长的拥堵时间
    test_hull = {
        "cluster_id": 0,
        "cluster_size": 8,
        "time_center": 480,  # 8:00 (峰值)
        "time_range": [360, 600],  # 6:00-10:00 (4小时拥堵)
        "center": [15.5, 480]
    }
    
    pattern = generator.calculate_congestion_pattern(test_hull, peak_congestion_km=8.0)
    
    print(f"✅ 测试数据:")
    print(f"  - 峰值时间: 8:00 (480分钟)")
    print(f"  - 拥堵时间段: 6:00-10:00 (360-600分钟)")  
    print(f"  - 峰值拥堵长度: 8km")
    print(f"  - 聚类大小: 8")
    
    # 显示拥堵分布
    print(f"\n拥堵时间分布模式:")
    for hour in range(5, 12):  # 显示5:00-11:00
        value = pattern[hour] if hour < len(pattern) else ''
        status = "🔴峰值" if hour == 8 else "🟡拥堵" if value and value != '' else "🟢畅通"
        print(f"  {hour:2d}:00 → {value:>3s}km {status}")
    
    print(f"\n三角形分布特征验证:")
    non_empty_count = sum(1 for v in pattern if v and v != '')
    peak_found = '8' in pattern[8] if len(pattern) > 8 else False
    edges_empty = (pattern[6] == '' and pattern[10] == '') if len(pattern) > 10 else False
    
    print(f"  ✅ 拥堵时段数量: {non_empty_count}")
    print(f"  ✅ 峰值时间正确: {peak_found}")
    print(f"  ✅ 边缘时间为空: {edges_empty}")

def test_multiple_scenarios():
    """测试多种拥堵场景"""
    print("\n🧪 测试多种拥堵场景...")
    
    generator = ExcelOutputGenerator()
    
    scenarios = [
        {
            "name": "早高峰短时拥堵",
            "hull": {
                "time_center": 450,  # 7:30
                "time_range": [420, 480],  # 7:00-8:00
                "cluster_size": 3
            },
            "peak_km": 3.0
        },
        {
            "name": "晚高峰长时拥堵", 
            "hull": {
                "time_center": 1080,  # 18:00
                "time_range": [1020, 1140],  # 17:00-19:00
                "cluster_size": 6
            },
            "peak_km": 6.0
        },
        {
            "name": "深夜轻微拥堵",
            "hull": {
                "time_center": 1440,  # 24:00 (第二天0:00)
                "time_range": [1410, 1470],  # 23:30-24:30
                "cluster_size": 2
            },
            "peak_km": 2.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        pattern = generator.calculate_congestion_pattern(scenario['hull'], scenario['peak_km'])
        
        # 找到非空时段
        congestion_hours = []
        for hour, value in enumerate(pattern):
            if value and value != '':
                congestion_hours.append((hour, value))
        
        if congestion_hours:
            print(f"拥堵时段: {len(congestion_hours)} 小时")
            for hour, value in congestion_hours:
                print(f"  {hour:2d}:00 → {value}km")
        else:
            print("无拥堵预测")

def main():
    """主测试函数"""
    print("🚀 开始测试三角形拥堵分布计算")
    print("=" * 50)
    
    try:
        # 测试详细的三角形分布
        test_detailed_triangle_distribution()
        
        # 测试多种场景
        test_multiple_scenarios()
        
        print("\n" + "=" * 50)
        print("✅ 三角形分布测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()











