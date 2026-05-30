#!/usr/bin/env python3
"""
测试方向感知的交通拥堵分析系统
"""

import os
import sys
sys.path.append('/home/dizhihuang/graduate/predict_workflow/version_1_1english')

from functions import (
    process_direction_aware_traffic_data,
    process_multiple_direction_aware_traffic_data,
    load_road_info,
    get_road_kp_range,
    get_kp_location_name,
    extract_road_type_from_filename
)

def test_road_info_loading():
    """测试道路信息加载功能"""
    print("🧪 测试道路信息加载功能...")
    
    # 测试関越道信息
    kannetsu_info = load_road_info("関越道")
    if not kannetsu_info.empty:
        print(f"✅ 関越道信息加载成功: {len(kannetsu_info)} 个KP点")
        kp_min, kp_max = get_road_kp_range("関越道", "上")
        print(f"   上り方向KP范围: {kp_min:.1f} - {kp_max:.1f} km")
        
        kp_min, kp_max = get_road_kp_range("関越道", "下")
        print(f"   下り方向KP范围: {kp_min:.1f} - {kp_max:.1f} km")
    else:
        print("❌ 関越道信息加载失败")
    
    # 测试外環道信息
    gaikan_info = load_road_info("外環道")
    if not gaikan_info.empty:
        print(f"✅ 外環道信息加载成功: {len(gaikan_info)} 个KP点")
        kp_min, kp_max = get_road_kp_range("外環道", "上")
        print(f"   上り方向KP范围: {kp_min:.1f} - {kp_max:.1f} km")
    else:
        print("❌ 外環道信息加载失败")
    
    # 测试KP位置名称获取
    test_kp = 30.0
    location_name = get_kp_location_name("関越道", test_kp, "上")
    print(f"✅ KP{test_kp}对应区间: {location_name}")
    

def test_direction_aware_processing():
    """测试方向感知的数据处理"""
    print("\n🧪 测试方向感知的数据处理...")
    
    # 查找可用的関越道数据文件
    data_dir = "/home/dizhihuang/graduate/predict_workflow/data/processed_data"
    
    test_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if "関越道" in file and file.endswith('.csv'):
                test_files.append(os.path.join(root, file))
                if len(test_files) >= 2:  # 只取前两个文件进行测试
                    break
        if test_files:
            break
    
    if not test_files:
        print("❌ 未找到関越道测试数据文件")
        return
    
    print(f"📁 找到测试文件: {len(test_files)} 个")
    for i, file in enumerate(test_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # 测试单文件上り方向处理
    print("\n📊 测试单文件上り方向处理...")
    try:
        triangles_up, fig_path_up = process_direction_aware_traffic_data(
            test_files[0], "上", "test_output"
        )
        print(f"✅ 上り方向处理成功: {len(triangles_up)} 个多边形")
        print(f"   可视化图片: {fig_path_up}")
    except Exception as e:
        print(f"❌ 上り方向处理失败: {e}")
    
    # 测试单文件下り方向处理
    print("\n📊 测试单文件下り方向处理...")
    try:
        triangles_down, fig_path_down = process_direction_aware_traffic_data(
            test_files[0], "下", "test_output"
        )
        print(f"✅ 下り方向处理成功: {len(triangles_down)} 个多边形")
        print(f"   可视化图片: {fig_path_down}")
    except Exception as e:
        print(f"❌ 下り方向处理失败: {e}")
    
    # 测试多文件处理
    if len(test_files) > 1:
        print("\n📊 测试多文件上り方向处理...")
        try:
            triangles_multi, fig_path_multi = process_multiple_direction_aware_traffic_data(
                test_files[:2], "上", "test_output"
            )
            print(f"✅ 多文件上り方向处理成功: {len(triangles_multi)} 个多边形")
            print(f"   合并可视化图片: {fig_path_multi}")
        except Exception as e:
            print(f"❌ 多文件上り方向处理失败: {e}")


def test_road_type_extraction():
    """测试道路类型提取"""
    print("\n🧪 测试道路类型提取...")
    
    test_files = [
        "関越道_上_2024_04-05.csv",
        "外環道_下_2023_03-15.csv",
        "東名道_上_2022_08-10.csv",
        "unknown_road_2024.csv"
    ]
    
    for file in test_files:
        road_type = extract_road_type_from_filename(file)
        print(f"   {file} → {road_type}")


def main():
    """主测试函数"""
    print("🚀 开始测试方向感知的交通拥堵分析系统")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs("test_output", exist_ok=True)
    
    try:
        # 测试道路信息加载
        test_road_info_loading()
        
        # 测试道路类型提取
        test_road_type_extraction()
        
        # 测试方向感知处理
        test_direction_aware_processing()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("📁 输出文件保存在 test_output/ 目录")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()