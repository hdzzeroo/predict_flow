#!/usr/bin/env python3
"""
Improved F1 Score calculation for hotspot validation
改进的F1分数计算方法
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class ImprovedF1Calculator:
    """改进的F1分数计算器"""
    
    def __init__(self):
        self.grid_resolution = 1.0  # KP网格分辨率（km）
        self.time_resolution = 60   # 时间网格分辨率（分钟）
    
    def method1_event_based_f1(self, actual_events: List[Dict], 
                              predicted_hotspots: List[Dict]) -> Tuple[float, float, float]:
        """
        方法1: 基于事件的F1计算
        将每个实际拥堵事件作为一个分类样本
        """
        print("📊 Method 1: Event-based F1 calculation")
        
        true_positives = 0    # 被正确预测的拥堵事件
        false_negatives = 0   # 未被预测到的拥堵事件
        false_positives = 0   # 需要通过热点有效性计算
        
        # 对每个实际事件进行分类
        covered_events = []
        uncovered_events = []
        
        for event in actual_events:
            is_covered = self._is_event_covered_by_hotspots(event, predicted_hotspots)
            
            if is_covered:
                true_positives += 1
                covered_events.append(event)
            else:
                false_negatives += 1
                uncovered_events.append(event)
        
        # 计算False Positives: 没有实际拥堵事件的热点区域
        # 这里使用"热点有效性"的概念
        effective_hotspots = 0
        for hotspot in predicted_hotspots:
            events_in_hotspot = sum(1 for event in actual_events 
                                  if self._is_event_in_hotspot(event, hotspot))
            if events_in_hotspot > 0:
                effective_hotspots += 1
        
        false_positives = len(predicted_hotspots) - effective_hotspots
        
        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   True Positives (covered events): {true_positives}")
        print(f"   False Negatives (missed events): {false_negatives}")
        print(f"   False Positives (ineffective hotspots): {false_positives}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1_score:.3f}")
        
        return precision, recall, f1_score
    
    def method2_grid_based_f1(self, actual_events: List[Dict], 
                             predicted_hotspots: List[Dict],
                             kp_range: Tuple[float, float] = (0, 120),
                             time_range: Tuple[float, float] = (0, 1440)) -> Tuple[float, float, float]:
        """
        方法2: 基于网格的F1计算
        将时空区域划分为网格，每个网格作为一个分类样本
        """
        print("📊 Method 2: Grid-based F1 calculation")
        
        # 创建时空网格
        kp_grid = np.arange(kp_range[0], kp_range[1], self.grid_resolution)
        time_grid = np.arange(time_range[0], time_range[1], self.time_resolution)
        
        total_grids = len(kp_grid) * len(time_grid)
        print(f"   Created {len(kp_grid)} x {len(time_grid)} = {total_grids} grid cells")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # 对每个网格单元进行分类
        for kp in kp_grid:
            for time_min in time_grid:
                # 检查网格是否有实际拥堵事件
                has_actual_event = self._grid_has_actual_event(
                    kp, time_min, actual_events
                )
                
                # 检查网格是否被热点覆盖
                is_predicted_hotspot = self._grid_in_predicted_hotspots(
                    kp, time_min, predicted_hotspots
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
        
        print(f"   True Positives: {true_positives}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        print(f"   True Negatives: {true_negatives}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1_score:.3f}")
        
        return precision, recall, f1_score
    
    def method3_weighted_f1(self, actual_events: List[Dict], 
                           predicted_hotspots: List[Dict]) -> Tuple[float, float, float]:
        """
        方法3: 基于权重的F1计算
        考虑热点区域的大小和重要性
        """
        print("📊 Method 3: Weighted F1 calculation")
        
        # 计算加权的TP、FP、FN
        weighted_tp = 0.0
        weighted_fp = 0.0
        weighted_fn = 0.0
        
        # 对每个热点区域计算权重分数
        for hotspot in predicted_hotspots:
            hotspot_area = hotspot.get('area', 1.0)
            hotspot_weight = np.log(hotspot_area + 1)  # 对数权重，避免过大
            
            # 计算该热点区域内的实际事件数
            events_in_hotspot = sum(1 for event in actual_events 
                                  if self._is_event_in_hotspot(event, hotspot))
            
            if events_in_hotspot > 0:
                # 有效热点：贡献加权TP
                weighted_tp += hotspot_weight
            else:
                # 无效热点：贡献加权FP
                weighted_fp += hotspot_weight
        
        # 计算未被覆盖事件的权重FN
        for event in actual_events:
            is_covered = self._is_event_covered_by_hotspots(event, predicted_hotspots)
            if not is_covered:
                # 事件权重可以基于拥堵严重程度
                event_weight = event.get('jam_length', 1.0) * event.get('duration_minutes', 60) / 60
                weighted_fn += event_weight
        
        # 计算加权指标
        precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0
        recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Weighted TP: {weighted_tp:.2f}")
        print(f"   Weighted FP: {weighted_fp:.2f}")
        print(f"   Weighted FN: {weighted_fn:.2f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1_score:.3f}")
        
        return precision, recall, f1_score
    
    def _is_event_covered_by_hotspots(self, event: Dict, hotspots: List[Dict]) -> bool:
        """检查事件是否被任何热点覆盖"""
        for hotspot in hotspots:
            if self._is_event_in_hotspot(event, hotspot):
                return True
        return False
    
    def _is_event_in_hotspot(self, event: Dict, hotspot: Dict) -> bool:
        """检查事件是否在热点区域内"""
        # 空间重叠检查
        spatial_overlap = self._has_overlap(
            event['kp_start'], event['kp_end'],
            hotspot['kp_start'], hotspot['kp_end']
        )
        
        # 时间重叠检查
        temporal_overlap = self._has_overlap(
            event['start_time_minutes'], event['end_time_minutes'],
            hotspot.get('time_start', 0), hotspot.get('time_end', 1440)
        )
        
        return spatial_overlap and temporal_overlap
    
    def _grid_has_actual_event(self, kp: float, time_min: float, events: List[Dict]) -> bool:
        """检查网格是否有实际拥堵事件"""
        kp_end = kp + self.grid_resolution
        time_end = time_min + self.time_resolution
        
        for event in events:
            if (self._has_overlap(kp, kp_end, event['kp_start'], event['kp_end']) and
                self._has_overlap(time_min, time_end, 
                                event['start_time_minutes'], event['end_time_minutes'])):
                return True
        return False
    
    def _grid_in_predicted_hotspots(self, kp: float, time_min: float, hotspots: List[Dict]) -> bool:
        """检查网格是否在预测热点区域内"""
        kp_end = kp + self.grid_resolution
        time_end = time_min + self.time_resolution
        
        for hotspot in hotspots:
            if (self._has_overlap(kp, kp_end, hotspot['kp_start'], hotspot['kp_end']) and
                self._has_overlap(time_min, time_end, 
                                hotspot.get('time_start', 0), hotspot.get('time_end', 1440))):
                return True
        return False
    
    def _has_overlap(self, a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        """检查两个区间是否有重叠"""
        return not (a_end <= b_start or b_end <= a_start)


def demonstrate_f1_methods():
    """演示三种F1计算方法"""
    print("🧪 Demonstrating different F1 calculation methods")
    print("=" * 80)
    
    # 模拟数据
    actual_events = [
        {'kp_start': 20, 'kp_end': 25, 'start_time_minutes': 480, 'end_time_minutes': 540, 'jam_length': 5, 'duration_minutes': 60},
        {'kp_start': 30, 'kp_end': 35, 'start_time_minutes': 1020, 'end_time_minutes': 1080, 'jam_length': 5, 'duration_minutes': 60},
        {'kp_start': 50, 'kp_end': 55, 'start_time_minutes': 600, 'end_time_minutes': 660, 'jam_length': 5, 'duration_minutes': 60},
    ]
    
    predicted_hotspots = [
        {'kp_start': 18, 'kp_end': 28, 'time_start': 420, 'time_end': 600, 'area': 1000, 'cluster_id': 0},
        {'kp_start': 45, 'kp_end': 60, 'time_start': 540, 'time_end': 720, 'area': 1500, 'cluster_id': 1},
        {'kp_start': 70, 'kp_end': 80, 'time_start': 480, 'time_end': 600, 'area': 800, 'cluster_id': 2},  # 无效热点
    ]
    
    calculator = ImprovedF1Calculator()
    
    print("\\n🎯 Comparison of F1 calculation methods:")
    print("-" * 80)
    
    # 方法1: 基于事件
    p1, r1, f1_1 = calculator.method1_event_based_f1(actual_events, predicted_hotspots)
    print()
    
    # 方法2: 基于网格 (使用较小的范围进行演示)
    p2, r2, f1_2 = calculator.method2_grid_based_f1(
        actual_events, predicted_hotspots, 
        kp_range=(0, 100), time_range=(400, 1200)
    )
    print()
    
    # 方法3: 基于权重
    p3, r3, f1_3 = calculator.method3_weighted_f1(actual_events, predicted_hotspots)
    print()
    
    # 总结比较
    print("📊 Summary Comparison:")
    print(f"   Method 1 (Event-based):  P={p1:.3f}, R={r1:.3f}, F1={f1_1:.3f}")
    print(f"   Method 2 (Grid-based):   P={p2:.3f}, R={r2:.3f}, F1={f1_2:.3f}") 
    print(f"   Method 3 (Weighted):     P={p3:.3f}, R={r3:.3f}, F1={f1_3:.3f}")
    
    print("\\n💡 Method Recommendations:")
    print("   - Method 1: Simple and intuitive, good for basic validation")
    print("   - Method 2: Comprehensive coverage analysis, computationally intensive")
    print("   - Method 3: Considers importance and scale, most realistic for traffic applications")


if __name__ == "__main__":
    demonstrate_f1_methods()