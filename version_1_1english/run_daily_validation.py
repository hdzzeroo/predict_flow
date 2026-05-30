#!/usr/bin/env python3
"""
Run Daily Validation Script
运行按天验证的脚本
"""

import sys
import os
from daily_validation_system import DailyHotspotValidationSystem

def main():
    """主函数"""
    print("🚀 Daily Traffic Congestion Hotspot Validation")
    print("=" * 80)
    
    # 检查验证数据文件
    csv_file = "may_2025_actual_data.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Validation data file not found: {csv_file}")
        print("💡 Please ensure you have the actual data CSV file")
        return False
    
    print(f"✅ Found validation data: {csv_file}")
    
    # 创建验证系统
    validator = DailyHotspotValidationSystem()
    
    try:
        print("\n🔬 Starting daily validation process...")
        print("⏳ This may take some time as we generate predictions for each day...")
        
        # 运行5月份的按天验证
        results = validator.run_monthly_validation(
            csv_path=csv_file,
            target_month=5,
            target_year=2025
        )
        
        if not results:
            print("❌ Validation failed - no results generated")
            return False
        
        print("\n" + "=" * 80)
        print("🎉 Daily Validation Completed Successfully!")
        print("=" * 80)
        
        # 显示整体结果摘要
        print("\n📊 Monthly Summary:")
        print(f"   📅 Validation Period: {results['validation_period']}")
        print(f"   📈 Total Days: {results['total_days']}")
        print(f"   📍 Total Events: {results['total_actual_events']}")
        print(f"   🎯 Overall Coverage Rate: {results['overall_coverage_rate']:.1%}")
        print(f"   ⚡ Average F1 Score: {results['avg_f1_score']:.3f}")
        print(f"   📊 Average Spatial Coverage: {results['avg_spatial_coverage_rate']:.1%}")
        print(f"   ⏰ Average Temporal Accuracy: {results['avg_temporal_accuracy_rate']:.1%}")
        
        # 显示性能等级
        overall_score = (results['avg_spatial_coverage_rate'] + 
                        results['avg_temporal_accuracy_rate'] + 
                        results['avg_f1_score']) / 3
        
        if overall_score >= 0.8:
            grade = "A (Excellent)"
            comment = "🌟 Your system shows excellent daily prediction capability!"
        elif overall_score >= 0.7:
            grade = "B (Good)"
            comment = "👍 Your system performs well across different days."
        elif overall_score >= 0.6:
            grade = "C (Fair)"
            comment = "⚠️ Your system shows moderate daily performance, consider improvements."
        else:
            grade = "D (Needs Improvement)"
            comment = "🔧 Your system needs significant improvements for daily accuracy."
        
        print(f"   🏆 Overall Performance: {overall_score:.1%} - Grade {grade}")
        print(f"   💬 {comment}")
        
        # 显示最佳和最差天数
        best_day = max(results['daily_results'], key=lambda x: x.f1_score)
        worst_day = min(results['daily_results'], key=lambda x: x.f1_score)
        
        print(f"\n🥇 Best Day: {best_day.date} (F1: {best_day.f1_score:.3f}, Events: {best_day.total_actual_events})")
        print(f"🥉 Most Challenging Day: {worst_day.date} (F1: {worst_day.f1_score:.3f}, Events: {worst_day.total_actual_events})")
        
        # 生成详细报告
        print("\n📄 Generating detailed reports...")
        report_file = validator.generate_daily_validation_report()
        
        print(f"\n📁 Generated files:")
        print(f"   📋 {report_file} - Detailed daily validation report")
        
        # 显示每日结果概览
        print("\n📅 Daily Results Overview:")
        print("   Date       | Events | Hotspots | Coverage | F1 Score")
        print("   -----------|--------|----------|----------|----------")
        
        for result in results['daily_results']:
            print(f"   {result.date} |   {result.total_actual_events:2d}   |    {result.predicted_hotspots:2d}    |  {result.spatial_coverage_rate:5.1%}  |  {result.f1_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎊 Daily validation completed successfully!")
        print("💡 Check the generated report for detailed analysis of daily performance patterns.")
    else:
        print("\n💡 Please check the error messages above and try again.")