#!/usr/bin/env python3
"""
Test script for grid-based validation method
测试基于网格的验证方法
"""

import json
import os
import sys
from validation_system import HotspotValidationSystem

def load_workflow_result():
    """加载现有的工作流结果"""
    result_file = "output/workflow_state.json"
    
    if not os.path.exists(result_file):
        print(f"❌ Workflow result file not found: {result_file}")
        print("💡 Please run the traffic analysis system first:")
        print("   python test_complete_workflow.py")
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            workflow_result = json.load(f)
        
        print(f"✅ Loaded workflow result from: {result_file}")
        print(f"   Triangles: {len(workflow_result.get('triangles', []))}")
        print(f"   Clusters: {len(workflow_result.get('clusters', []))}")
        print(f"   Hotspots: {len(workflow_result.get('hulls', []))}")
        
        return workflow_result
    
    except Exception as e:
        print(f"❌ Error loading workflow result: {str(e)}")
        return None

def add_time_ranges_to_hotspots(workflow_result):
    """为热点添加时间范围信息"""
    hulls = workflow_result.get('hulls', [])
    triangles = workflow_result.get('triangles', [])
    clusters = workflow_result.get('clusters', [])
    
    print("🔧 Adding time range information to hotspots...")
    
    # 为每个hull添加时间范围
    for i, hull in enumerate(hulls):
        cluster_id = hull['cluster_id']
        
        # 找到对应的聚类
        if cluster_id < len(clusters):
            cluster_triangles = clusters[cluster_id]
            
            # 收集该聚类中所有三角形的时间信息
            cluster_times = []
            for triangle_idx in cluster_triangles:
                if triangle_idx < len(triangles):
                    triangle = triangles[triangle_idx]
                    # 假设center[1]是时间坐标（分钟）
                    time_coord = triangle.get('center', [0, 0])[1]
                    cluster_times.append(time_coord)
            
            if cluster_times:
                time_start = min(cluster_times) - 60  # 扩展1小时边界
                time_end = max(cluster_times) + 60
                hull['time_range'] = [max(0, time_start), min(1440, time_end)]
                print(f"   Hotspot {cluster_id}: time range {time_start:.0f}-{time_end:.0f} minutes")
            else:
                hull['time_range'] = [0, 1440]  # 默认全天
                print(f"   Hotspot {cluster_id}: using default time range (全天)")
    
    return workflow_result

def create_sample_may_events():
    """创建基于实际系统输出的样本事件"""
    print("🎯 Creating sample May 2025 events based on system output...")
    
    # 基于你的系统输出，创建一些逼真的5月份事件
    sample_events = [
        # 基于hotspot cluster 2 (KP 72.5-108.2)
        {'id': 0, 'date': '2025-05-04', 'kp_start': 75.0, 'kp_end': 80.0, 
         'start_time_minutes': 900, 'end_time_minutes': 960, 'jam_length': 5.0, 'duration_minutes': 60},
        {'id': 1, 'date': '2025-05-08', 'kp_start': 95.0, 'kp_end': 100.0, 
         'start_time_minutes': 1000, 'end_time_minutes': 1080, 'jam_length': 5.0, 'duration_minutes': 80},
        
        # 基于hotspot cluster 1 (KP 26.7-61.4)  
        {'id': 2, 'date': '2025-05-12', 'kp_start': 30.0, 'kp_end': 35.0, 
         'start_time_minutes': 1140, 'end_time_minutes': 1200, 'jam_length': 5.0, 'duration_minutes': 60},
        {'id': 3, 'date': '2025-05-15', 'kp_start': 50.0, 'kp_end': 55.0, 
         'start_time_minutes': 1200, 'end_time_minutes': 1280, 'jam_length': 5.0, 'duration_minutes': 80},
        
        # 基于hotspot cluster 0 (KP -2.8-30.8)
        {'id': 4, 'date': '2025-05-18', 'kp_start': 10.0, 'kp_end': 15.0, 
         'start_time_minutes': 920, 'end_time_minutes': 980, 'jam_length': 5.0, 'duration_minutes': 60},
        {'id': 5, 'date': '2025-05-22', 'kp_start': 25.0, 'kp_end': 30.0, 
         'start_time_minutes': 1000, 'end_time_minutes': 1070, 'jam_length': 5.0, 'duration_minutes': 70},
        
        # 一些不在热点区域的事件（测试FN）
        {'id': 6, 'date': '2025-05-25', 'kp_start': 110.0, 'kp_end': 115.0, 
         'start_time_minutes': 800, 'end_time_minutes': 860, 'jam_length': 5.0, 'duration_minutes': 60},
        {'id': 7, 'date': '2025-05-28', 'kp_start': 15.0, 'kp_end': 20.0, 
         'start_time_minutes': 400, 'end_time_minutes': 460, 'jam_length': 5.0, 'duration_minutes': 60},
    ]
    
    print(f"✅ Created {len(sample_events)} sample events")
    
    # 显示事件统计
    hotspot_events = 0
    non_hotspot_events = 0
    
    for event in sample_events:
        if event['kp_start'] >= 70:  # cluster 2 区域
            hotspot_events += 1
        elif 25 <= event['kp_start'] <= 65:  # cluster 1 区域  
            hotspot_events += 1
        elif event['kp_start'] <= 35:  # cluster 0 区域
            hotspot_events += 1
        else:
            non_hotspot_events += 1
    
    print(f"   预期在热点区域内: {hotspot_events} 个事件")
    print(f"   预期在热点区域外: {non_hotspot_events} 个事件")
    
    return sample_events

def run_grid_validation_test():
    """运行基于网格的验证测试"""
    print("🧪 Testing Grid-Based Validation Method")
    print("=" * 80)
    
    # 1. 加载工作流结果
    workflow_result = load_workflow_result()
    if not workflow_result:
        return False
    
    # 2. 添加时间范围信息
    workflow_result = add_time_ranges_to_hotspots(workflow_result)
    
    # 3. 创建样本事件数据
    sample_events = create_sample_may_events()
    
    # 4. 初始化验证系统
    validator = HotspotValidationSystem()
    validator.actual_events = sample_events
    validator.predicted_hotspots = validator.load_predicted_hotspots(workflow_result)
    
    print("\n" + "=" * 80)
    print("🔬 Running Grid-Based Validation")
    print("=" * 80)
    
    # 5. 运行各项验证
    print("\n1️⃣ Spatial Coverage Validation")
    print("-" * 50)
    spatial_coverage = validator.validate_spatial_coverage()
    
    print("\n2️⃣ Temporal Accuracy Validation") 
    print("-" * 50)
    temporal_accuracy = validator.validate_temporal_accuracy()
    
    print("\n3️⃣ Hotspot Density Analysis")
    print("-" * 50)
    hotspot_densities = validator.calculate_hotspot_densities()
    
    print("\n4️⃣ Grid-Based F1 Score Calculation")
    print("-" * 50)
    precision, recall, f1_score = validator.calculate_precision_recall()
    
    # 6. 生成综合报告
    print("\n" + "=" * 80)
    print("📊 Comprehensive Validation Results")
    print("=" * 80)
    
    print(f"\n🎯 Key Performance Indicators:")
    print(f"   Spatial Coverage Rate:  {spatial_coverage:.1%}")
    print(f"   Temporal Accuracy Rate: {temporal_accuracy:.1%}")
    print(f"   Precision (Grid-based): {precision:.3f}")
    print(f"   Recall (Grid-based):    {recall:.3f}")
    print(f"   F1 Score (Grid-based):  {f1_score:.3f}")
    
    # 计算综合评分
    overall_score = (spatial_coverage + temporal_accuracy + f1_score) / 3
    
    if overall_score >= 0.8:
        grade = "A (Excellent)"
        comment = "🌟 Your system demonstrates excellent hotspot identification capability!"
    elif overall_score >= 0.7:
        grade = "B (Good)"
        comment = "👍 Your system shows good performance in hotspot identification."
    elif overall_score >= 0.6:
        grade = "C (Fair)"
        comment = "⚠️ Your system shows moderate performance, consider parameter optimization."
    else:
        grade = "D (Needs Improvement)"
        comment = "🔧 Your system needs improvements for better accuracy."
    
    print(f"\n📈 Overall Performance: {overall_score:.1%} - Grade {grade}")
    print(f"💬 {comment}")
    
    # 7. 热点密度分析
    if hotspot_densities:
        avg_density = sum(hotspot_densities) / len(hotspot_densities)
        print(f"\n🏠 Hotspot Density Analysis:")
        print(f"   Average Density: {avg_density:.3f} events per area unit")
        print(f"   Highest Density: {max(hotspot_densities):.3f}")
        print(f"   Lowest Density:  {min(hotspot_densities):.3f}")
    
    print("\n🎊 Grid-based validation completed successfully!")
    print("✨ This method provides the most comprehensive and accurate assessment.")
    
    return True

def main():
    """主函数"""
    success = run_grid_validation_test()
    
    if success:
        print("\n💡 Next Steps:")
        print("   1. Fine-tune grid resolution for different accuracy levels")
        print("   2. Run validation with real May 2025 data when available")
        print("   3. Compare results with other validation methods")
        print("   4. Use insights to optimize system parameters")
    else:
        print("\n❌ Validation test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()