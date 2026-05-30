#!/usr/bin/env python3
"""
Daily Traffic Congestion Hotspot Validation System
按天为单位的拥堵热点验证系统
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass
from implementation import compiled_agent

@dataclass
class DailyValidationMetrics:
    """每日验证指标数据类"""
    date: str                           # 验证日期
    spatial_coverage_rate: float        # 空间覆盖率
    temporal_accuracy_rate: float       # 时间准确率  
    precision: float                    # 精确率
    recall: float                      # 召回率
    f1_score: float                    # F1分数
    total_actual_events: int           # 实际拥堵事件总数
    covered_events: int                # 被覆盖的事件数
    predicted_hotspots: int            # 预测热点数

class DailyHotspotValidationSystem:
    """按天为单位的拥堵热点验证系统"""
    
    def __init__(self):
        self.daily_results = []            # 每日验证结果
        self.validation_summary = None     # 整体验证摘要
    
    def load_actual_data_by_date(self, csv_path: str) -> Dict[str, List[Dict]]:
        """
        按日期加载实际拥堵数据
        """
        print(f"📊 Loading daily actual congestion data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            print(f"✅ Loaded {len(df)} total events")
            
            # 按日期分组
            events_by_date = {}
            for _, row in df.iterrows():
                date = str(row['date'])[:10]  # 取YYYY-MM-DD部分
                
                if date not in events_by_date:
                    events_by_date[date] = []
                
                # 转换时间格式
                start_time_str = str(row['発生時刻'])
                if ':' in start_time_str:
                    time_parts = start_time_str.split(':')
                    start_time_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                else:
                    start_time_minutes = 480  # 默认8:00
                
                # 创建事件记录
                event = {
                    'date': date,
                    'kp_start': float(row.get('発生Ｋｐ', 0)),
                    'jam_length': float(row.get('ピーク長', 1.0)),
                    'start_time_minutes': start_time_minutes,
                    'duration_minutes': int(row.get('渋滞時間', 60)),
                    'road_type': str(row.get('道路番号', '関越道'))
                }
                event['kp_end'] = event['kp_start'] + event['jam_length']
                event['end_time_minutes'] = event['start_time_minutes'] + event['duration_minutes']
                
                events_by_date[date].append(event)
            
            print(f"✅ Grouped events by date: {len(events_by_date)} days")
            for date, events in events_by_date.items():
                print(f"   {date}: {len(events)} events")
            
            return events_by_date
            
        except Exception as e:
            print(f"❌ Error loading actual data: {str(e)}")
            return {}
    
    def generate_daily_prediction(self, date: str) -> Dict[str, Any]:
        """
        为指定日期生成拥堵热点预测
        """
        print(f"🔮 Generating prediction for {date}")
        
        try:
            # 解析日期
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            month = date_obj.month
            day = date_obj.day
            
            # 构建查询语句
            user_input = f"analyze congestion patterns for Kan-Etsu Expressway on {date_obj.strftime('%Y/%m/%d')}"
            print(f"   Query: {user_input}")
            
            # 运行工作流预测
            result = compiled_agent.invoke({"user_input": user_input})
            
            # 提取预测热点
            hulls = result.get('hulls', [])
            predicted_hotspots = []
            
            for hull in hulls:
                hotspot = {
                    'cluster_id': hull['cluster_id'],
                    'cluster_size': hull['cluster_size'],
                    'kp_start': hull['kp_range'][0],
                    'kp_end': hull['kp_range'][1],
                    'time_start': hull.get('time_range', [0, 1440])[0],
                    'time_end': hull.get('time_range', [0, 1440])[1],
                    'area': hull['area']
                }
                predicted_hotspots.append(hotspot)
            
            print(f"   ✅ Generated {len(predicted_hotspots)} hotspot predictions")
            return {
                'date': date,
                'predicted_hotspots': predicted_hotspots,
                'workflow_result': result
            }
            
        except Exception as e:
            print(f"   ❌ Error generating prediction for {date}: {str(e)}")
            return {
                'date': date,
                'predicted_hotspots': [],
                'workflow_result': {}
            }
    
    def validate_single_day(self, date: str, actual_events: List[Dict], predicted_hotspots: List[Dict]) -> DailyValidationMetrics:
        """
        验证单天的预测结果
        """
        print(f"🔬 Validating {date}: {len(actual_events)} actual events vs {len(predicted_hotspots)} predictions")
        
        if not actual_events:
            return DailyValidationMetrics(
                date=date,
                spatial_coverage_rate=0.0,
                temporal_accuracy_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_actual_events=0,
                covered_events=0,
                predicted_hotspots=len(predicted_hotspots)
            )
        
        # 空间覆盖率验证
        covered_events = 0
        time_accurate_events = 0
        
        for event in actual_events:
            # 检查空间覆盖
            spatial_covered = False
            temporal_covered = False
            
            for hotspot in predicted_hotspots:
                # 空间重叠检查
                if (event['kp_start'] <= hotspot['kp_end'] and 
                    event['kp_end'] >= hotspot['kp_start']):
                    spatial_covered = True
                    
                    # 时间重叠检查
                    if (event['start_time_minutes'] <= hotspot['time_end'] and 
                        event['end_time_minutes'] >= hotspot['time_start']):
                        temporal_covered = True
                        break
            
            if spatial_covered:
                covered_events += 1
            if temporal_covered:
                time_accurate_events += 1
        
        # 计算基本指标
        spatial_coverage_rate = covered_events / len(actual_events) if actual_events else 0
        temporal_accuracy_rate = time_accurate_events / len(actual_events) if actual_events else 0
        
        # 网格验证计算精确率、召回率、F1分数
        precision, recall, f1_score = self._calculate_grid_metrics(actual_events, predicted_hotspots)
        
        return DailyValidationMetrics(
            date=date,
            spatial_coverage_rate=spatial_coverage_rate,
            temporal_accuracy_rate=temporal_accuracy_rate,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_actual_events=len(actual_events),
            covered_events=covered_events,
            predicted_hotspots=len(predicted_hotspots)
        )
    
    def _calculate_grid_metrics(self, actual_events: List[Dict], predicted_hotspots: List[Dict]) -> Tuple[float, float, float]:
        """
        使用网格方法计算精确率、召回率、F1分数
        """
        if not actual_events or not predicted_hotspots:
            return 0.0, 0.0, 0.0
        
        # 创建网格
        kp_min = min(min(event['kp_start'] for event in actual_events), 
                     min(hotspot['kp_start'] for hotspot in predicted_hotspots))
        kp_max = max(max(event['kp_end'] for event in actual_events),
                     max(hotspot['kp_end'] for hotspot in predicted_hotspots))
        
        grid_size_kp = 2.0  # 2km网格
        grid_size_time = 60  # 60分钟网格
        
        kp_grids = int((kp_max - kp_min) / grid_size_kp) + 1
        time_grids = 24  # 24小时
        
        # 初始化网格
        actual_grid = np.zeros((kp_grids, time_grids))
        predicted_grid = np.zeros((kp_grids, time_grids))
        
        # 填充实际事件网格
        for event in actual_events:
            kp_idx = int((event['kp_start'] - kp_min) / grid_size_kp)
            time_idx = int(event['start_time_minutes'] / 60)
            if 0 <= kp_idx < kp_grids and 0 <= time_idx < time_grids:
                actual_grid[kp_idx, time_idx] = 1
        
        # 填充预测热点网格
        for hotspot in predicted_hotspots:
            for kp in np.arange(hotspot['kp_start'], hotspot['kp_end'], grid_size_kp):
                for time_min in range(int(hotspot['time_start']), int(hotspot['time_end']), grid_size_time):
                    kp_idx = int((kp - kp_min) / grid_size_kp)
                    time_idx = int(time_min / 60)
                    if 0 <= kp_idx < kp_grids and 0 <= time_idx < time_grids:
                        predicted_grid[kp_idx, time_idx] = 1
        
        # 计算混淆矩阵
        tp = np.sum((actual_grid == 1) & (predicted_grid == 1))
        fp = np.sum((actual_grid == 0) & (predicted_grid == 1))
        fn = np.sum((actual_grid == 1) & (predicted_grid == 0))
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def run_monthly_validation(self, csv_path: str, target_month: int = 5, target_year: int = 2025) -> Dict[str, Any]:
        """
        运行整月的按天验证
        """
        print(f"🚀 Starting monthly validation for {target_year}-{target_month:02d}")
        print("=" * 80)
        
        # 加载按日期分组的实际数据
        events_by_date = self.load_actual_data_by_date(csv_path)
        
        if not events_by_date:
            print("❌ No actual data found")
            return {}
        
        # 获取目标月份的所有日期
        target_dates = [date for date in events_by_date.keys() 
                       if date.startswith(f"{target_year}-{target_month:02d}")]
        target_dates.sort()
        
        print(f"🎯 Found {len(target_dates)} days with actual data in target month")
        
        daily_results = []
        
        # 逐天验证
        for date in target_dates:
            print(f"\n📅 Processing {date}...")
            
            # 生成当天预测
            prediction_result = self.generate_daily_prediction(date)
            
            # 验证当天结果
            actual_events = events_by_date[date]
            predicted_hotspots = prediction_result['predicted_hotspots']
            
            daily_metrics = self.validate_single_day(date, actual_events, predicted_hotspots)
            daily_results.append(daily_metrics)
            
            print(f"   📊 Results: Coverage={daily_metrics.spatial_coverage_rate:.1%}, "
                  f"F1={daily_metrics.f1_score:.3f}, Events={daily_metrics.total_actual_events}")
        
        # 计算整体统计
        self.daily_results = daily_results
        summary = self._generate_monthly_summary(daily_results)
        
        return summary
    
    def _generate_monthly_summary(self, daily_results: List[DailyValidationMetrics]) -> Dict[str, Any]:
        """
        生成月度验证摘要
        """
        if not daily_results:
            return {}
        
        # 计算平均指标
        avg_spatial_coverage = np.mean([r.spatial_coverage_rate for r in daily_results])
        avg_temporal_accuracy = np.mean([r.temporal_accuracy_rate for r in daily_results])
        avg_precision = np.mean([r.precision for r in daily_results])
        avg_recall = np.mean([r.recall for r in daily_results])
        avg_f1_score = np.mean([r.f1_score for r in daily_results])
        
        total_actual_events = sum(r.total_actual_events for r in daily_results)
        total_covered_events = sum(r.covered_events for r in daily_results)
        total_predicted_hotspots = sum(r.predicted_hotspots for r in daily_results)
        
        summary = {
            'validation_period': f"{daily_results[0].date} to {daily_results[-1].date}",
            'total_days': len(daily_results),
            'avg_spatial_coverage_rate': avg_spatial_coverage,
            'avg_temporal_accuracy_rate': avg_temporal_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1_score,
            'total_actual_events': total_actual_events,
            'total_covered_events': total_covered_events,
            'total_predicted_hotspots': total_predicted_hotspots,
            'overall_coverage_rate': total_covered_events / total_actual_events if total_actual_events > 0 else 0,
            'daily_results': daily_results
        }
        
        self.validation_summary = summary
        return summary
    
    def generate_daily_validation_report(self, output_file: str = "daily_validation_report.md"):
        """
        生成按天验证的详细报告
        """
        if not self.validation_summary:
            print("❌ No validation results to report")
            return
        
        summary = self.validation_summary
        
        report_lines = [
            "# Daily Traffic Congestion Hotspot Validation Report",
            f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Validation Period**: {summary['validation_period']}",
            f"**Total Days Validated**: {summary['total_days']}",
            "",
            "## Executive Summary",
            f"- **Total Actual Events**: {summary['total_actual_events']}",
            f"- **Total Covered Events**: {summary['total_covered_events']}",
            f"- **Overall Coverage Rate**: {summary['overall_coverage_rate']:.1%}",
            f"- **Average Spatial Coverage**: {summary['avg_spatial_coverage_rate']:.1%}",
            f"- **Average Temporal Accuracy**: {summary['avg_temporal_accuracy_rate']:.1%}",
            f"- **Average Precision**: {summary['avg_precision']:.3f}",
            f"- **Average Recall**: {summary['avg_recall']:.3f}",
            f"- **Average F1 Score**: {summary['avg_f1_score']:.3f}",
            "",
            "## Daily Validation Results",
            "",
            "| Date | Events | Hotspots | Coverage | Precision | Recall | F1 Score |",
            "|------|--------|----------|----------|-----------|--------|----------|"
        ]
        
        # 添加每日结果
        for result in summary['daily_results']:
            report_lines.append(
                f"| {result.date} | {result.total_actual_events} | {result.predicted_hotspots} | "
                f"{result.spatial_coverage_rate:.1%} | {result.precision:.3f} | "
                f"{result.recall:.3f} | {result.f1_score:.3f} |"
            )
        
        # 添加分析和建议
        report_lines.extend([
            "",
            "## Performance Analysis",
            "",
            f"**Best Performing Day**: {max(summary['daily_results'], key=lambda x: x.f1_score).date} "
            f"(F1: {max(summary['daily_results'], key=lambda x: x.f1_score).f1_score:.3f})",
            "",
            f"**Most Challenging Day**: {min(summary['daily_results'], key=lambda x: x.f1_score).date} "
            f"(F1: {min(summary['daily_results'], key=lambda x: x.f1_score).f1_score:.3f})",
            "",
            "## Recommendations",
            "",
            "1. **Daily Pattern Analysis**: The system shows varying performance across different days",
            "2. **Parameter Optimization**: Consider day-specific parameter tuning",
            "3. **Data Quality**: Ensure consistent data quality across all validation days",
            "",
            "---",
            "*Report generated by Daily Hotspot Validation System*"
        ])
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Daily validation report saved to: {output_file}")
        
        return output_file

if __name__ == "__main__":
    # 测试运行
    validator = DailyHotspotValidationSystem()
    
    # 运行5月份的按天验证
    results = validator.run_monthly_validation("may_2025_actual_data.csv", target_month=5)
    
    if results:
        # 生成报告
        validator.generate_daily_validation_report()
        
        print("\n" + "=" * 80)
        print("📊 Monthly Validation Summary")
        print("=" * 80)
        print(f"Overall Coverage Rate: {results['overall_coverage_rate']:.1%}")
        print(f"Average F1 Score: {results['avg_f1_score']:.3f}")
        print(f"Total Days: {results['total_days']}")
        print(f"Total Events: {results['total_actual_events']}")