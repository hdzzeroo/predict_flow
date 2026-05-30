#!/usr/bin/env python3
"""
Traffic Congestion Hotspot Validation System
用于验证拥堵热点识别系统的准确性
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass

@dataclass
class ValidationMetrics:
    """验证指标数据类"""
    spatial_coverage_rate: float    # 空间覆盖率
    temporal_accuracy_rate: float   # 时间准确率  
    precision: float                # 精确率
    recall: float                   # 召回率
    f1_score: float                 # F1分数
    hotspot_densities: List[float]  # 热点密度列表
    total_actual_events: int        # 实际拥堵事件总数
    covered_events: int             # 被覆盖的事件数

class HotspotValidationSystem:
    """拥堵热点验证系统"""
    
    def __init__(self):
        self.actual_events = []        # 实际拥堵事件
        self.predicted_hotspots = []   # 系统识别的热点区域
        self.validation_results = None # 验证结果
    
    def load_actual_data_from_excel(self, excel_path: str, month: int = 5, 
                                   road_type: str = "関越道") -> List[Dict]:
        """
        从Excel文件加载2025年5月的实际拥堵数据
        """
        print(f"📊 Loading actual congestion data from: {excel_path}")
        
        try:
            # 由于Excel读取问题，我们先尝试模拟数据结构
            # 实际使用时需要根据Excel具体格式调整
            print("⚠️ Note: Due to library issues, using simulated data structure")
            print("🔧 You need to manually extract May 2025 data and convert to CSV format")
            
            # 模拟数据结构 - 实际使用时替换为真实数据
            sample_events = self._generate_sample_actual_events()
            
            print(f"✅ Loaded {len(sample_events)} actual congestion events")
            return sample_events
            
        except Exception as e:
            print(f"❌ Error loading Excel data: {str(e)}")
            print("🔄 Using sample data for demonstration")
            return self._generate_sample_actual_events()
    
    def _generate_sample_actual_events(self) -> List[Dict]:
        """生成样本实际事件数据（用于演示）"""
        # 这里生成一些模拟的2025年5月拥堵事件
        np.random.seed(42)
        sample_events = []
        
        # 模拟在不同KP和时间段的拥堵事件
        for i in range(20):  # 生成20个样本事件
            event = {
                'id': i,
                'date': f'2025-05-{np.random.randint(1, 31):02d}',
                'kp_start': np.random.uniform(0, 80),
                'jam_length': np.random.uniform(1, 5),
                'start_time_minutes': np.random.randint(360, 1200),  # 6:00-20:00
                'duration_minutes': np.random.randint(30, 180),
                'road_type': '関越道'
            }
            event['kp_end'] = event['kp_start'] + event['jam_length']
            event['end_time_minutes'] = event['start_time_minutes'] + event['duration_minutes']
            sample_events.append(event)
        
        return sample_events
    
    def load_predicted_hotspots(self, workflow_result: Dict[str, Any]) -> List[Dict]:
        """
        从工作流结果中加载系统识别的热点区域
        """
        print("🎯 Loading predicted hotspots from system output")
        
        hulls = workflow_result.get('hulls', [])
        
        predicted_hotspots = []
        for hull in hulls:
            hotspot = {
                'cluster_id': hull['cluster_id'],
                'cluster_size': hull['cluster_size'],
                'kp_start': hull['kp_range'][0],
                'kp_end': hull['kp_range'][1],
                'time_start': hull.get('time_range', [0, 1440])[0],  # 默认全天
                'time_end': hull.get('time_range', [0, 1440])[1],
                'area': hull['area']
            }
            predicted_hotspots.append(hotspot)
        
        print(f"✅ Loaded {len(predicted_hotspots)} predicted hotspots")
        return predicted_hotspots
    
    def validate_spatial_coverage(self) -> float:
        """验证空间覆盖率"""
        print("📍 Validating spatial coverage...")
        
        covered_events = 0
        
        for event in self.actual_events:
            event_kp_start = event['kp_start']
            event_kp_end = event['kp_end']
            
            # 检查是否与任何热点区域有重叠
            for hotspot in self.predicted_hotspots:
                hotspot_kp_start = hotspot['kp_start']
                hotspot_kp_end = hotspot['kp_end']
                
                # 检查空间重叠
                if self._has_spatial_overlap(event_kp_start, event_kp_end, 
                                           hotspot_kp_start, hotspot_kp_end):
                    covered_events += 1
                    break
        
        coverage_rate = covered_events / len(self.actual_events) if self.actual_events else 0
        print(f"   Spatial coverage rate: {coverage_rate:.2%}")
        return coverage_rate
    
    def validate_temporal_accuracy(self) -> float:
        """验证时间准确率"""
        print("⏰ Validating temporal accuracy...")
        
        time_matched_events = 0
        
        for event in self.actual_events:
            event_time_start = event['start_time_minutes']
            event_time_end = event['end_time_minutes']
            
            # 检查是否与任何热点的时间段有重叠
            for hotspot in self.predicted_hotspots:
                hotspot_time_start = hotspot['time_start']
                hotspot_time_end = hotspot['time_end']
                
                # 检查时间重叠
                if self._has_temporal_overlap(event_time_start, event_time_end,
                                            hotspot_time_start, hotspot_time_end):
                    time_matched_events += 1
                    break
        
        time_accuracy = time_matched_events / len(self.actual_events) if self.actual_events else 0
        print(f"   Temporal accuracy rate: {time_accuracy:.2%}")
        return time_accuracy
    
    def calculate_hotspot_densities(self) -> List[float]:
        """计算每个热点区域的拥堵事件密度"""
        print("📈 Calculating hotspot densities...")
        
        densities = []
        
        for hotspot in self.predicted_hotspots:
            events_in_hotspot = 0
            hotspot_kp_start = hotspot['kp_start']
            hotspot_kp_end = hotspot['kp_end']
            hotspot_time_start = hotspot['time_start']
            hotspot_time_end = hotspot['time_end']
            
            # 计算落在该热点区域内的实际事件数
            for event in self.actual_events:
                if (self._has_spatial_overlap(event['kp_start'], event['kp_end'],
                                            hotspot_kp_start, hotspot_kp_end) and
                    self._has_temporal_overlap(event['start_time_minutes'], event['end_time_minutes'],
                                             hotspot_time_start, hotspot_time_end)):
                    events_in_hotspot += 1
            
            # 计算密度 (事件数 / 区域面积)
            hotspot_area = max(hotspot.get('area', 1), 1)  # 避免除零
            density = events_in_hotspot / hotspot_area * 1000  # 标准化
            densities.append(density)
            
            print(f"   Hotspot {hotspot['cluster_id']}: {events_in_hotspot} events, "
                  f"density = {density:.3f}")
        
        return densities
    
    def calculate_precision_recall(self) -> Tuple[float, float, float]:
        """使用基于网格的方法计算精确率、召回率和F1分数"""
        print("🎯 Calculating precision, recall, and F1 score using grid-based method...")
        
        # 网格参数设置
        kp_resolution = 2.0      # KP网格分辨率: 1km
        time_resolution = 60     # 时间网格分辨率: 60分钟
        
        # 确定网格范围
        all_kps = []
        all_times = []
        
        # 从实际事件中获取范围
        for event in self.actual_events:
            all_kps.extend([event['kp_start'], event['kp_end']])
            all_times.extend([event['start_time_minutes'], event['end_time_minutes']])
        
        # 从热点区域中获取范围
        for hotspot in self.predicted_hotspots:
            all_kps.extend([hotspot['kp_start'], hotspot['kp_end']])
            all_times.extend([hotspot['time_start'], hotspot['time_end']])
        
        if not all_kps or not all_times:
            print("   ⚠️ No data available for grid-based calculation")
            return 0.0, 0.0, 0.0
        
        # 设置网格范围（稍微扩展边界）
        kp_min = max(0, min(all_kps) - 5)
        kp_max = max(all_kps) + 5
        time_min = max(0, min(all_times) - 120)
        time_max = min(1440, max(all_times) + 120)
        
        # 创建网格
        kp_grid = np.arange(kp_min, kp_max, kp_resolution)
        time_grid = np.arange(time_min, time_max, time_resolution)
        
        total_grids = len(kp_grid) * len(time_grid)
        print(f"   📊 Created {len(kp_grid)} x {len(time_grid)} = {total_grids} grid cells")
        print(f"   📍 KP range: {kp_min:.1f} - {kp_max:.1f} km")
        print(f"   ⏰ Time range: {time_min:.0f} - {time_max:.0f} minutes")
        
        # 初始化计数器
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # 对每个网格单元进行分类
        for kp in kp_grid:
            for time_min_cell in time_grid:
                # 检查网格是否有实际拥堵事件
                has_actual_event = self._grid_has_actual_event(
                    kp, time_min_cell, kp_resolution, time_resolution
                )
                
                # 检查网格是否被热点覆盖
                is_predicted_hotspot = self._grid_in_predicted_hotspots(
                    kp, time_min_cell, kp_resolution, time_resolution
                )
                
                # 分类统计
                if has_actual_event and is_predicted_hotspot:
                    true_positives += 1
                elif has_actual_event and not is_predicted_hotspot:
                    false_negatives += 1
                elif not has_actual_event and is_predicted_hotspot:
                    false_positives += 1
                else:  # not has_actual_event and not is_predicted_hotspot
                    true_negatives += 1
        
        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   📈 Grid Classification Results:")
        print(f"      True Positives (TP): {true_positives}")
        print(f"      False Positives (FP): {false_positives}")
        print(f"      False Negatives (FN): {false_negatives}")
        print(f"      True Negatives (TN): {true_negatives}")
        print(f"   🎯 Performance Metrics:")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall: {recall:.3f}")
        print(f"      F1 Score: {f1_score:.3f}")
        
        return precision, recall, f1_score
    
    def _grid_has_actual_event(self, kp: float, time_min: float, 
                              kp_resolution: float, time_resolution: float) -> bool:
        """检查网格是否有实际拥堵事件"""
        kp_end = kp + kp_resolution
        time_end = time_min + time_resolution
        
        for event in self.actual_events:
            if (self._has_overlap(kp, kp_end, event['kp_start'], event['kp_end']) and
                self._has_overlap(time_min, time_end, 
                                event['start_time_minutes'], event['end_time_minutes'])):
                return True
        return False
    
    def _grid_in_predicted_hotspots(self, kp: float, time_min: float,
                                   kp_resolution: float, time_resolution: float) -> bool:
        """检查网格是否在预测热点区域内"""
        kp_end = kp + kp_resolution
        time_end = time_min + time_resolution
        
        for hotspot in self.predicted_hotspots:
            if (self._has_overlap(kp, kp_end, hotspot['kp_start'], hotspot['kp_end']) and
                self._has_overlap(time_min, time_end, 
                                hotspot.get('time_start', 0), hotspot.get('time_end', 1440))):
                return True
        return False

    def _has_spatial_overlap(self, a_start: float, a_end: float, 
                           b_start: float, b_end: float) -> bool:
        """检查两个空间区间是否有重叠"""
        return not (a_end <= b_start or b_end <= a_start)
    
    def _has_temporal_overlap(self, a_start: float, a_end: float,
                            b_start: float, b_end: float) -> bool:
        """检查两个时间区间是否有重叠"""
        return not (a_end <= b_start or b_end <= a_start)
    
    def _has_overlap(self, a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        """通用区间重叠检查方法"""
        return not (a_end <= b_start or b_end <= a_start)
    
    def run_comprehensive_validation(self, excel_path: str, 
                                   workflow_result: Dict[str, Any]) -> ValidationMetrics:
        """运行综合验证"""
        print("🚀 Starting comprehensive validation")
        print("=" * 80)
        
        # 1. 加载数据
        self.actual_events = self.load_actual_data_from_excel(excel_path)
        self.predicted_hotspots = self.load_predicted_hotspots(workflow_result)
        
        if not self.actual_events or not self.predicted_hotspots:
            print("❌ Missing data for validation")
            return None
        
        print("-" * 80)
        
        # 2. 执行各项验证
        spatial_coverage = self.validate_spatial_coverage()
        temporal_accuracy = self.validate_temporal_accuracy()
        hotspot_densities = self.calculate_hotspot_densities()
        precision, recall, f1_score = self.calculate_precision_recall()
        
        # 3. 汇总结果
        covered_events = sum(1 for event in self.actual_events 
                           if any(self._has_spatial_overlap(event['kp_start'], event['kp_end'],
                                                           hotspot['kp_start'], hotspot['kp_end'])
                                 for hotspot in self.predicted_hotspots))
        
        metrics = ValidationMetrics(
            spatial_coverage_rate=spatial_coverage,
            temporal_accuracy_rate=temporal_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            hotspot_densities=hotspot_densities,
            total_actual_events=len(self.actual_events),
            covered_events=covered_events
        )
        
        self.validation_results = metrics
        return metrics
    
    def generate_validation_report(self, output_path: str = "validation_report.md") -> str:
        """生成验证报告"""
        if not self.validation_results:
            print("❌ No validation results available")
            return ""
        
        print(f"📄 Generating validation report: {output_path}")
        
        metrics = self.validation_results
        report_sections = []
        
        # 标题和概要
        report_sections.extend([
            "# Traffic Congestion Hotspot Validation Report",
            f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Validation Target**: May 2025 Actual Congestion Events",
            "",
            "## Executive Summary",
            f"- **Total Actual Events**: {metrics.total_actual_events}",
            f"- **Covered Events**: {metrics.covered_events}",
            f"- **Spatial Coverage Rate**: {metrics.spatial_coverage_rate:.2%}",
            f"- **Temporal Accuracy Rate**: {metrics.temporal_accuracy_rate:.2%}",
            f"- **System Precision**: {metrics.precision:.3f}",
            f"- **System Recall**: {metrics.recall:.3f}",
            f"- **F1 Score**: {metrics.f1_score:.3f}",
            ""
        ])
        
        # 详细分析
        report_sections.extend([
            "## Detailed Analysis",
            "",
            "### 1. Spatial Coverage Analysis",
            f"The system's hotspot regions covered **{metrics.spatial_coverage_rate:.1%}** of actual congestion events.",
            f"This means {metrics.covered_events} out of {metrics.total_actual_events} real congestion events "
            "occurred within the predicted hotspot areas.",
            "",
            "### 2. Temporal Accuracy Analysis", 
            f"**{metrics.temporal_accuracy_rate:.1%}** of actual congestion events occurred during the time periods "
            "identified by the system as congestion-prone.",
            "",
            "### 3. Grid-Based F1 Score Analysis",
            f"Using comprehensive grid-based validation method:",
            f"- **Precision ({metrics.precision:.3f})**: Proportion of predicted hotspot grids that actually had congestion",
            f"- **Recall ({metrics.recall:.3f})**: Proportion of actual congestion grids that were correctly predicted",
            f"- **F1 Score ({metrics.f1_score:.3f})**: Harmonic mean of precision and recall",
            "",
            "**Grid-based Method Advantages:**",
            "- Comprehensive coverage of entire spatio-temporal space",
            "- Each grid cell classified as TP, FP, FN, or TN",
            "- More accurate than event-based or simple overlap methods",
            "- Provides detailed spatial distribution of prediction accuracy",
            ""
        ])
        
        # 热点密度分析
        if metrics.hotspot_densities:
            avg_density = np.mean(metrics.hotspot_densities)
            report_sections.extend([
                "### 4. Hotspot Density Analysis",
                f"- **Average Hotspot Density**: {avg_density:.3f} events per area unit",
                f"- **Highest Density**: {max(metrics.hotspot_densities):.3f}",
                f"- **Lowest Density**: {min(metrics.hotspot_densities):.3f}",
                ""
            ])
        
        # 结论和建议
        overall_performance = (metrics.spatial_coverage_rate + metrics.temporal_accuracy_rate + metrics.f1_score) / 3
        
        if overall_performance >= 0.8:
            performance_level = "Excellent"
            recommendation = "The system shows high accuracy in hotspot identification. Consider deploying for operational use."
        elif overall_performance >= 0.6:
            performance_level = "Good"
            recommendation = "The system performs well but could benefit from parameter tuning and additional training data."
        elif overall_performance >= 0.4:
            performance_level = "Fair"
            recommendation = "The system shows promise but requires significant improvements before deployment."
        else:
            performance_level = "Poor"
            recommendation = "The system needs major revisions to improve hotspot identification accuracy."
        
        report_sections.extend([
            "## Conclusions and Recommendations",
            f"**Overall Performance Level**: {performance_level} ({overall_performance:.1%})",
            "",
            f"**Recommendation**: {recommendation}",
            "",
            "### Key Findings:",
            f"1. The system successfully identified hotspot areas that covered {metrics.spatial_coverage_rate:.1%} of actual congestion events",
            f"2. Time-based predictions achieved {metrics.temporal_accuracy_rate:.1%} accuracy",
            f"3. The F1 score of {metrics.f1_score:.3f} indicates {'good' if metrics.f1_score >= 0.6 else 'moderate'} overall system performance",
            "",
            "---",
            "*Report generated by Traffic Congestion Hotspot Validation System*"
        ])
        
        report_content = "\\n".join(report_sections)
        
        # 保存报告
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ Validation report saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {str(e)}")
        
        return report_content
    
    def plot_validation_results(self, output_path: str = "validation_plot.png"):
        """绘制验证结果可视化图表"""
        if not self.validation_results:
            print("❌ No validation results to plot")
            return
        
        print(f"📊 Creating validation visualization: {output_path}")
        
        metrics = self.validation_results
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traffic Congestion Hotspot Validation Results', fontsize=16, fontweight='bold')
        
        # 1. 覆盖率对比
        categories = ['Spatial Coverage', 'Temporal Accuracy', 'Precision', 'Recall']
        values = [metrics.spatial_coverage_rate, metrics.temporal_accuracy_rate, 
                 metrics.precision, metrics.recall]
        
        bars = ax1.bar(categories, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.set_title('Validation Metrics Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. F1分数和综合性能
        performance_metrics = ['F1 Score', 'Overall Performance']
        overall_perf = (metrics.spatial_coverage_rate + metrics.temporal_accuracy_rate + metrics.f1_score) / 3
        perf_values = [metrics.f1_score, overall_perf]
        
        bars2 = ax2.bar(performance_metrics, perf_values, color=['gold', 'lightcoral'])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title('System Performance Summary')
        
        for bar, value in zip(bars2, perf_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 热点密度分布
        if metrics.hotspot_densities:
            ax3.hist(metrics.hotspot_densities, bins=max(5, len(metrics.hotspot_densities)//2), 
                    color='lightblue', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Density (events per area unit)')
            ax3.set_ylabel('Number of Hotspots')
            ax3.set_title('Hotspot Density Distribution')
        else:
            ax3.text(0.5, 0.5, 'No density data available', ha='center', va='center', 
                    transform=ax3.transAxes)
            ax3.set_title('Hotspot Density Distribution')
        
        # 4. 事件覆盖情况
        labels = ['Covered Events', 'Uncovered Events']
        sizes = [metrics.covered_events, metrics.total_actual_events - metrics.covered_events]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax4.set_title('Actual Events Coverage')
        
        plt.tight_layout()
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Validation plot saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to save plot: {str(e)}")
        
        plt.close()


def main():
    """主函数 - 演示验证系统使用"""
    print("🔬 Traffic Congestion Hotspot Validation System")
    print("=" * 80)
    
    # 初始化验证系统
    validator = HotspotValidationSystem()
    
    # 模拟工作流结果 (实际使用时从系统获取)
    sample_workflow_result = {
        'hulls': [
            {
                'cluster_id': 0,
                'cluster_size': 8,
                'kp_range': [10.0, 30.0],
                'time_range': [480, 600],  # 8:00-10:00
                'area': 1000.0
            },
            {
                'cluster_id': 1,
                'cluster_size': 5,
                'kp_range': [40.0, 60.0],
                'time_range': [1020, 1140], # 17:00-19:00
                'area': 800.0
            }
        ]
    }
    
    # 运行验证 (Excel路径需要根据实际情况调整)
    excel_path = "/path/to/your/2025_data.xlsx"  # 替换为实际路径
    
    try:
        metrics = validator.run_comprehensive_validation(excel_path, sample_workflow_result)
        
        if metrics:
            print("\\n" + "=" * 80)
            print("🎉 Validation completed successfully!")
            
            # 生成报告和图表
            validator.generate_validation_report("hotspot_validation_report.md")
            validator.plot_validation_results("hotspot_validation_plot.png")
            
            print("\\n📋 Validation Summary:")
            print(f"   Spatial Coverage: {metrics.spatial_coverage_rate:.2%}")
            print(f"   Temporal Accuracy: {metrics.temporal_accuracy_rate:.2%}")
            print(f"   F1 Score: {metrics.f1_score:.3f}")
        else:
            print("❌ Validation failed")
            
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()