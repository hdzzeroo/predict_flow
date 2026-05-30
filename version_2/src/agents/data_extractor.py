"""
Data Extraction Agent for historical traffic data processing.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

import os
import pandas as pd
from datetime import datetime, date, time
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from .base_agent import BaseAgent, monitor_performance
from ..core.data_structures import (
    TrafficEvent, Triangle, HistoricalData, SystemConfig
)
from ..core.exceptions import DataValidationError, FileNotFoundError


class DataExtractionAgent(BaseAgent):
    """数据提取Agent - 负责历史数据提取和预处理"""

    def __init__(self, config: SystemConfig):
        super().__init__(config, "DataExtractionAgent")

    def validate_input(self, input_data: Dict) -> bool:
        """验证输入数据"""
        required_fields = ['target_date', 'road_name']
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"缺少必要字段: {field}")
                return False

        # 验证日期格式
        try:
            datetime.strptime(input_data['target_date'], '%Y-%m-%d')
        except ValueError:
            self.logger.error("日期格式必须是YYYY-MM-DD")
            return False

        return True

    @monitor_performance
    def _process_internal(self, input_data: Dict) -> Dict:
        """
        提取历史数据并生成三种格式的数据

        Returns:
            {
                'raw_data': pd.DataFrame,           # 原始CSV数据
                'triangles': List[Triangle],        # 三角形数据
                'visualization_data': Dict,         # 可视化所需数据
                'statistics': Dict                  # 统计信息
            }
        """
        try:
            # 1. 加载历史数据（3年同期数据）
            raw_data = self._load_historical_data(
                target_date=input_data['target_date'],
                road_name=input_data['road_name'],
                direction=input_data.get('direction')
            )

            if raw_data.empty:
                raise DataValidationError(f"未找到相关历史数据: {input_data}")

            # 2. 转换为三角形数据
            triangles = self._convert_to_triangles(raw_data)

            if not triangles:
                raise DataValidationError("无法从历史数据生成有效的三角形")

            # 3. 生成可视化数据
            viz_data = self._prepare_visualization_data(triangles)

            # 4. 计算统计信息
            statistics = self._calculate_statistics(raw_data, triangles)

            return {
                'raw_data': raw_data,
                'triangles': triangles,
                'visualization_data': viz_data,
                'statistics': statistics
            }

        except Exception as e:
            self.logger.error(f"数据提取失败: {e}")
            raise

    @monitor_performance
    def _load_historical_data(self, target_date: str, road_name: str, direction: str = None) -> pd.DataFrame:
        """加载历史数据 - 基于DATA_SPECIFICATION.md的文件命名规则"""
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        historical_files = []

        # 查找过去3年同期数据
        for year_offset in [1, 2, 3]:
            try:
                historical_date = target_date_obj.replace(year=target_date_obj.year - year_offset)
            except ValueError:
                # 处理闰年2月29日的情况
                historical_date = target_date_obj.replace(year=target_date_obj.year - year_offset, day=28)

            # 根据DATA_SPECIFICATION.md的命名规则
            if direction:
                pattern = f"{road_name}_{direction}_{historical_date.strftime('%Y_%m-%d')}.csv"
                file_path = os.path.join(self.config.data_path, pattern)
                if os.path.exists(file_path):
                    historical_files.append(file_path)
            else:
                # 如果未指定方向，查找两个方向的数据
                for dir in ['上', '下']:
                    pattern = f"{road_name}_{dir}_{historical_date.strftime('%Y_%m-%d')}.csv"
                    file_path = os.path.join(self.config.data_path, pattern)
                    if os.path.exists(file_path):
                        historical_files.append(file_path)

        if not historical_files:
            raise FileNotFoundError(f"未找到{road_name}在{target_date}的历史数据文件")

        # 合并所有历史数据
        all_data = []
        for file_path in historical_files:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
                self.logger.info(f"成功加载文件: {file_path}, 记录数: {len(df)}")
            except Exception as e:
                self.logger.warning(f"读取文件失败 {file_path}: {e}")

        if not all_data:
            raise DataValidationError("所有历史数据文件都无法读取")

        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"合并历史数据完成，总记录数: {len(combined_data)}")
        return combined_data

    @monitor_performance
    def _convert_to_triangles(self, raw_data: pd.DataFrame) -> List[Triangle]:
        """将原始数据转换为三角形数据"""
        triangles = []

        for idx, row in raw_data.iterrows():
            try:
                # 创建TrafficEvent
                traffic_event = self._create_traffic_event(row)
                traffic_event.validate()

                # 生成三角形
                triangle = self._create_triangle_from_event(traffic_event, str(idx))
                triangle.validate()

                triangles.append(triangle)

            except Exception as e:
                self.logger.warning(f"跳过无效数据行 {idx}: {e}")

        self.logger.info(f"生成三角形数量: {len(triangles)}")
        return triangles

    def _create_traffic_event(self, row: pd.Series) -> TrafficEvent:
        """从数据行创建TrafficEvent"""
        return TrafficEvent(
            date=datetime.strptime(str(row['date']), '%Y-%m-%d').date(),
            direction=str(row['上下']),
            cause=str(row['原因']),
            road_name=str(row['道路番号']),
            start_time=datetime.strptime(str(row['発生時刻']), '%H:%M:%S').time(),
            peak_time=datetime.strptime(str(row['ピーク時刻']), '%H:%M:%S').time(),
            peak_length=float(row['ピーク長']),
            start_kp=float(row['発生Ｋｐ']),
            initial_length=float(row['発生時渋滞長']),
            duration=int(row['渋滞時間'])
        )

    def _create_triangle_from_event(self, event: TrafficEvent, triangle_id: str) -> Triangle:
        """从TrafficEvent创建Triangle"""
        # 时间转换为分钟（从0:00开始）
        start_minutes = event.start_time.hour * 60 + event.start_time.minute
        peak_minutes = event.peak_time.hour * 60 + event.peak_time.minute

        # 计算三角形顶点坐标
        # 顶点：峰值时间和峰值拥堵的中心位置
        apex_x = event.start_kp + event.peak_length / 2
        apex_y = peak_minutes

        # 底边：拥堵开始和结束的位置
        base_start_x = event.start_kp
        base_end_x = event.start_kp + event.peak_length
        base_y = start_minutes

        # 三角形的三个顶点
        vertices = [
            (apex_x, apex_y),           # 顶点
            (base_start_x, base_y),     # 底边左端点
            (base_end_x, base_y)        # 底边右端点
        ]

        # 计算中心点
        center_x = (apex_x + base_start_x + base_end_x) / 3
        center_y = (apex_y + base_y + base_y) / 3

        # 计算面积（三角形面积公式）
        area = 0.5 * abs((base_end_x - base_start_x) * (apex_y - base_y))

        # 计算严重程度（基于面积和持续时间）
        severity = min(1.0, (area * event.duration) / 10000)  # 归一化处理

        return Triangle(
            id=triangle_id,
            vertices=vertices,
            center=(center_x, center_y),
            area=area,
            kp_start=base_start_x,
            kp_end=base_end_x,
            time_start=start_minutes,
            time_peak=peak_minutes,
            duration=event.duration,
            severity=severity,
            direction=event.direction,
            road_name=event.road_name,
            source_event=event
        )

    def _prepare_visualization_data(self, triangles: List[Triangle]) -> Dict:
        """准备可视化所需数据"""
        if not triangles:
            return {}

        return {
            'x_range': (
                min(t.kp_start for t in triangles) - 1,
                max(t.kp_end for t in triangles) + 1
            ),
            'y_range': (
                min(t.time_start for t in triangles) - 30,
                max(t.time_peak for t in triangles) + 30
            ),
            'triangle_count': len(triangles),
            'directions': list(set(t.direction for t in triangles)),
            'roads': list(set(t.road_name for t in triangles))
        }

    def _calculate_statistics(self, raw_data: pd.DataFrame, triangles: List[Triangle]) -> Dict:
        """计算统计信息"""
        if raw_data.empty or not triangles:
            return {}

        return {
            'total_events': len(raw_data),
            'total_triangles': len(triangles),
            'date_range': {
                'start': raw_data['date'].min(),
                'end': raw_data['date'].max()
            },
            'road_name': raw_data['道路番号'].iloc[0] if len(raw_data) > 0 else '',
            'avg_duration': np.mean([t.duration for t in triangles]),
            'avg_severity': np.mean([t.severity for t in triangles]),
            'peak_times': {
                'earliest': min(t.time_peak for t in triangles),
                'latest': max(t.time_peak for t in triangles)
            },
            'location_range': {
                'start_kp': min(t.kp_start for t in triangles),
                'end_kp': max(t.kp_end for t in triangles)
            },
            'directions': list(set(t.direction for t in triangles))
        }