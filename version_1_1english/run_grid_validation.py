#!/usr/bin/env python3
"""
Quick Grid-Based Validation Runner
快速网格验证运行器
"""

import os
import json
from validation_system import HotspotValidationSystem

def main():
    """主函数 - 快速运行基于网格的验证"""
    print("🔬 Quick Grid-Based Validation")
    print("=" * 60)
    
    # 检查工作流结果文件
    workflow_file = "output/workflow_state.json"
    if not os.path.exists(workflow_file):
        print("❌ No workflow result found!")
        print("💡 Please run analysis first:")
        print("   python test_complete_workflow.py")
        return
    
    # 加载工作流结果
    with open(workflow_file, 'r', encoding='utf-8') as f:
        workflow_result = json.load(f)
    
    print(f"✅ Loaded system analysis results")
    print(f"   - Hotspots: {len(workflow_result.get('hulls', []))}")
    
    # 初始化验证系统
    validator = HotspotValidationSystem()
    
    # 使用模拟数据进行演示
    print("🎯 Using simulated validation data...")
    validator.actual_events = validator._generate_sample_actual_events()
    validator.predicted_hotspots = validator.load_predicted_hotspots(workflow_result)
    
    # 运行验证
    print("\n🔬 Running validation...")
    metrics = validator.run_comprehensive_validation("", workflow_result)
    
    if metrics:
        print("\n📊 Quick Summary:")
        print(f"   Grid-based F1 Score: {metrics.f1_score:.3f}")
        print(f"   Spatial Coverage:    {metrics.spatial_coverage_rate:.1%}")
        print(f"   Overall Performance: {(metrics.spatial_coverage_rate + metrics.temporal_accuracy_rate + metrics.f1_score)/3:.1%}")
        
        # 生成报告
        validator.generate_validation_report("quick_validation_report.md")
        print("\n📄 Detailed report saved: quick_validation_report.md")
        
        print("\n💡 For detailed analysis, see:")
        print("   - F1_VALIDATION_METHODS.md")  
        print("   - GRID_VALIDATION_ANALYSIS.md")
    else:
        print("❌ Validation failed")

if __name__ == "__main__":
    main()