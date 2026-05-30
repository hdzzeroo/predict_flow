"""
Core data structures for the multi-agent traffic prediction system.
Based on AI_DEVELOPMENT_GUIDE.md and DATA_SPECIFICATION.md
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from .exceptions import DataValidationError, TriangleValidationError


@dataclass
class TrafficEvent:
    """原始交通事件数据"""
    date: date
    direction: str  # "上" or "下"
    cause: str      # "交通集中"等
    road_name: str  # "東北道"
    start_time: time        # 発生時刻
    peak_time: time         # ピーク時刻
    peak_length: float      # ピーク長(km)
    start_kp: float         # 発生Ｋｐ
    initial_length: float   # 発生時渋滞長
    duration: int           # 渋滞時間(分钟)

    def validate(self) -> None:
        """数据验证 - 必须实现"""
        if self.peak_time < self.start_time:
            raise DataValidationError("峰值时间不能早于开始时间")
        if self.peak_length < self.initial_length:
            raise DataValidationError("峰值拥堵长度不能小于初始长度")
        if self.direction not in ["上", "下"]:
            raise DataValidationError(f"方向必须是'上'或'下'，当前值: {self.direction}")
        if self.duration <= 0:
            raise DataValidationError("拥堵时间必须大于0")
        if self.start_kp < 0:
            raise DataValidationError("起始KP不能为负数")
        if self.peak_length <= 0 or self.initial_length <= 0:
            raise DataValidationError("拥堵长度必须大于0")


@dataclass
class Triangle:
    """三角形数据结构"""
    id: str
    vertices: List[Tuple[float, float]]  # [(x1,y1), (x2,y2), (x3,y3)]
    center: Tuple[float, float]          # (center_x, center_y)
    area: float
    kp_start: float
    kp_end: float
    time_start: int      # 分钟，从0:00开始
    time_peak: int
    duration: int
    severity: float      # 0-1
    direction: str
    road_name: str
    source_event: TrafficEvent

    def validate(self) -> None:
        """验证三角形数据的物理合理性"""
        if len(self.vertices) != 3:
            raise TriangleValidationError("三角形必须有3个顶点")
        if self.area <= 0:
            raise TriangleValidationError("三角形面积必须为正数")
        if self.kp_end <= self.kp_start:
            raise TriangleValidationError("结束KP必须大于起始KP")
        if self.time_peak < self.time_start:
            raise TriangleValidationError("峰值时间不能早于开始时间")
        if self.time_start < 0 or self.time_peak < 0:
            raise TriangleValidationError("时间不能为负数")
        if self.severity < 0 or self.severity > 1:
            raise TriangleValidationError("严重程度必须在0-1之间")
        if self.direction not in ["上", "下"]:
            raise TriangleValidationError(f"方向必须是'上'或'下'，当前值: {self.direction}")


@dataclass
class PredictionResult:
    """预测结果"""
    prediction_id: str
    target_date: date
    direction: str
    predicted_triangles: List[Triangle]
    time_range: Tuple[time, time]  # (start, end)
    location_range: Tuple[float, float]  # (start_kp, end_kp)
    peak_time: time
    peak_location: float
    severity: str        # "high", "medium", "low"
    confidence: float    # 0-1
    explanation: str     # LLM生成的预测理由
    historical_basis: List[Triangle]  # 预测基于的历史三角形

    def validate(self) -> None:
        """验证预测结果"""
        if self.confidence < 0 or self.confidence > 1:
            raise DataValidationError("置信度必须在0-1之间")
        if self.severity not in ["high", "medium", "low"]:
            raise DataValidationError(f"严重程度必须是high/medium/low之一，当前值: {self.severity}")
        if self.direction not in ["上", "下"]:
            raise DataValidationError(f"方向必须是'上'或'下'，当前值: {self.direction}")
        if self.time_range[1] <= self.time_range[0]:
            raise DataValidationError("结束时间必须晚于开始时间")
        if self.location_range[1] <= self.location_range[0]:
            raise DataValidationError("结束位置必须大于开始位置")
        for triangle in self.predicted_triangles:
            triangle.validate()


@dataclass
class PredictionRequest:
    """预测请求"""
    target_date: str        # YYYY-MM-DD格式
    road_name: str         # 道路名称
    direction: Optional[str] = None  # 可选方向
    time_range: Optional[str] = None  # 可选时间范围 "17:00-19:00"

    def validate(self) -> None:
        """验证预测请求"""
        try:
            datetime.strptime(self.target_date, '%Y-%m-%d')
        except ValueError:
            raise DataValidationError("日期格式必须是YYYY-MM-DD")

        if self.direction and self.direction not in ["上", "下"]:
            raise DataValidationError(f"方向必须是'上'或'下'，当前值: {self.direction}")

        if self.time_range:
            try:
                parts = self.time_range.split('-')
                if len(parts) != 2:
                    raise ValueError
                datetime.strptime(parts[0].strip(), '%H:%M')
                datetime.strptime(parts[1].strip(), '%H:%M')
            except ValueError:
                raise DataValidationError("时间范围格式必须是HH:MM-HH:MM")


@dataclass
class HistoricalData:
    """历史数据包"""
    raw_data: pd.DataFrame           # 原始CSV数据
    triangles: List[Triangle]        # 三角形数据
    visualization_data: Dict         # 可视化所需数据
    statistics: Dict                 # 统计信息


@dataclass
class ValidationReport:
    """验证报告"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Agent响应基类"""
    success: bool
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0


# Configuration structures
@dataclass
class TriangleConfig:
    """三角形生成配置"""
    min_duration: int = 10           # 最小持续时间(分钟)
    min_length: float = 0.5          # 最小拥堵长度(km)
    time_resolution: int = 5         # 时间分辨率(分钟)
    merge_threshold: float = 0.8     # 合并阈值


@dataclass
class LLMConfig:
    """LLM配置"""
    model: str = "gpt-4-vision-preview"
    api_key: str = ""
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 30


@dataclass
class SystemConfig:
    """系统配置"""
    data_path: str
    output_path: str
    triangle_config: TriangleConfig = field(default_factory=TriangleConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    enable_logging: bool = True
    log_level: str = "INFO"