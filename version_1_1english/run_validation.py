#!/usr/bin/env python3
"""
Simple script to run hotspot validation
使用示例脚本
"""

import sys
import os
import json
from validation_system import HotspotValidationSystem

def load_workflow_result(result_file: str = "output/workflow_state.json") -> dict:
    """加载工作流结果"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading workflow result: {str(e)}")
        return {}

def extract_may_data_instructions():
    """打印提取5月数据的说明"""
    print("📋 Instructions for extracting May 2025 data:")
    print("-" * 60)
    print("1. Open your Excel file: ★2025_関東支社渋滞データ（01-05）SIC分割...")
    print("2. Filter data where '月' (Month) = 5")
    print("3. Filter data where '原因' (Cause) = '交通集中'")
    print("4. Filter data where '道路番号' (Road) = '関越道'")
    print("5. Save filtered data as CSV: 'may_2025_actual_data.csv'")
    print("6. Place the CSV file in the current directory")
    print()
    print("Expected CSV format:")
    print("date,原因,道路番号,発生時刻,ピーク時刻,ピーク長,発生Ｋｐ,発生時渋滞長,渋滞時間")
    print()

def main():
    """主函数"""
    print("🚀 Running Traffic Congestion Hotspot Validation")
    print("=" * 80)
    
    # 检查是否有工作流结果
    workflow_result_file = "output/workflow_state.json"
    if not os.path.exists(workflow_result_file):
        print(f"❌ Workflow result file not found: {workflow_result_file}")
        print("💡 Please run the traffic analysis system first to generate hotspot data")
        print("   Example: python test_complete_workflow.py")
        return False
    
    # 加载工作流结果
    print("📂 Loading system analysis results...")
    workflow_result = load_workflow_result(workflow_result_file)
    
    if not workflow_result or 'hulls' not in workflow_result:
        print("❌ Invalid workflow result - no hotspot data found")
        return False
    
    print(f"✅ Found {len(workflow_result['hulls'])} hotspots from system analysis")
    
    # 检查5月实际数据
    may_data_file = "may_2025_actual_data.csv"
    if not os.path.exists(may_data_file):
        print(f"❌ May 2025 actual data file not found: {may_data_file}")
        extract_may_data_instructions()
        print("⏳ Please prepare the data file and run this script again")
        return False
    
    # 运行验证
    print("🔬 Starting validation process...")
    print("-" * 80)
    
    validator = HotspotValidationSystem()
    
    try:
        # 运行综合验证
        metrics = validator.run_comprehensive_validation(may_data_file, workflow_result)
        
        if metrics:
            print("\\n" + "=" * 80)
            print("🎉 Validation completed successfully!")
            print("=" * 80)
            
            # 打印关键指标
            print("\\n📊 Key Validation Results:")
            print(f"   🎯 Spatial Coverage Rate: {metrics.spatial_coverage_rate:.1%}")
            print(f"      - {metrics.covered_events}/{metrics.total_actual_events} actual events covered by hotspots")
            print()
            print(f"   ⏰ Temporal Accuracy Rate: {metrics.temporal_accuracy_rate:.1%}")
            print(f"      - Time-based prediction accuracy")
            print()
            print(f"   🔍 System Performance:")
            print(f"      - Precision: {metrics.precision:.3f}")
            print(f"      - Recall: {metrics.recall:.3f}")
            print(f"      - F1 Score: {metrics.f1_score:.3f}")
            print()
            
            # 性能评估
            overall_score = (metrics.spatial_coverage_rate + 
                           metrics.temporal_accuracy_rate + 
                           metrics.f1_score) / 3
            
            if overall_score >= 0.8:
                grade = "A (Excellent)"
                comment = "🌟 Your system shows excellent hotspot identification capability!"
            elif overall_score >= 0.7:
                grade = "B (Good)"
                comment = "👍 Your system performs well in hotspot identification."
            elif overall_score >= 0.6:
                grade = "C (Fair)"
                comment = "⚠️ Your system shows moderate performance, consider improvements."
            else:
                grade = "D (Needs Improvement)"
                comment = "🔧 Your system needs significant improvements for better accuracy."
            
            print(f"   📈 Overall Performance: {overall_score:.1%} - Grade {grade}")
            print(f"   💬 {comment}")
            print()
            
            # 生成详细报告和图表
            print("📄 Generating detailed reports...")
            validator.generate_validation_report("hotspot_validation_report.md")
            validator.plot_validation_results("hotspot_validation_plot.png")
            
            print("\\n📁 Generated files:")
            print("   📋 hotspot_validation_report.md - Detailed validation report")
            print("   📊 hotspot_validation_plot.png - Validation results visualization")
            
            return True
        else:
            print("❌ Validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\\n🎊 Validation completed successfully! Check the generated reports for detailed analysis.")
    else:
        print("\\n💡 Please follow the instructions above to prepare your data and try again.")