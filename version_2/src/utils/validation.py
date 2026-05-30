"""
Validation utilities for the multi-agent traffic prediction system.
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime, time
import numpy as np

from ..core.data_structures import Triangle, PredictionResult, ValidationReport
from ..core.exceptions import DataValidationError


class ValidationUtils:
    """验证工具类"""

    @staticmethod
    def validate_triangle_data(triangle: Dict) -> bool:
        """验证三角形数据"""
        required_fields = ['vertices', 'center', 'area', 'kp_start', 'kp_end']
        if not all(field in triangle for field in required_fields):
            raise DataValidationError(f"三角形数据缺少必要字段: {required_fields}")

        # 物理约束验证
        if triangle['kp_end'] <= triangle['kp_start']:
            raise DataValidationError("KP结束位置必须大于起始位置")

        if triangle['area'] <= 0:
            raise DataValidationError("三角形面积必须为正数")

        return True

    @staticmethod
    def validate_physics_constraints(prediction: Dict) -> ValidationReport:
        """验证物理约束"""
        errors = []
        warnings = []

        triangle = prediction.get('predicted_triangle', {})

        # 时间约束检查
        try:
            apex_time = datetime.strptime(triangle['apex_time'], '%H:%M').time()
            base_start = datetime.strptime(triangle['base_start_time'], '%H:%M').time()
            base_end = datetime.strptime(triangle['base_end_time'], '%H:%M').time()

            if not (base_start <= apex_time <= base_end):
                errors.append("顶点时间必须在底边时间范围内")

        except (KeyError, ValueError) as e:
            errors.append(f"时间格式错误: {e}")

        # 空间约束检查
        try:
            apex_kp = triangle['apex_kp']
            base_start_kp = triangle['base_start_kp']
            base_end_kp = triangle['base_end_kp']

            if not (base_start_kp <= apex_kp <= base_end_kp):
                errors.append("顶点位置必须在底边位置范围内")

            if base_end_kp - base_start_kp > 50:  # 超过50km可能不合理
                warnings.append("预测拥堵范围超过50km，请确认合理性")

        except (KeyError, TypeError) as e:
            errors.append(f"位置数据错误: {e}")

        # 置信度检查
        confidence = triangle.get('confidence', 0)
        if not (0 <= confidence <= 1):
            errors.append("置信度必须在0-1之间")

        return ValidationReport(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def validate_historical_consistency(prediction: Dict, historical_data: Dict) -> ValidationReport:
        """验证历史一致性"""
        errors = []
        warnings = []

        try:
            predicted_triangle = prediction.get('predicted_triangle', {})
            historical_triangles = historical_data.get('triangles', [])

            if not historical_triangles:
                errors.append("缺少历史数据用于一致性验证")
                return ValidationReport(passed=False, errors=errors)

            # 时间范围一致性检查
            hist_time_ranges = [(t.time_start, t.time_peak) for t in historical_triangles]
            pred_start = ValidationUtils._time_to_minutes(predicted_triangle.get('base_start_time', ''))
            pred_peak = ValidationUtils._time_to_minutes(predicted_triangle.get('apex_time', ''))

            if pred_start is not None and pred_peak is not None:
                # 检查是否在历史时间范围的合理区间内
                hist_starts = [t[0] for t in hist_time_ranges]
                hist_peaks = [t[1] for t in hist_time_ranges]

                start_range = (min(hist_starts) - 60, max(hist_starts) + 60)  # ±1小时容忍度
                peak_range = (min(hist_peaks) - 60, max(hist_peaks) + 60)

                if not (start_range[0] <= pred_start <= start_range[1]):
                    warnings.append(f"预测开始时间 {pred_start} 超出历史范围 {start_range}")

                if not (peak_range[0] <= pred_peak <= peak_range[1]):
                    warnings.append(f"预测峰值时间 {pred_peak} 超出历史范围 {peak_range}")

            # 位置范围一致性检查
            hist_locations = [(t.kp_start, t.kp_end) for t in historical_triangles]
            pred_start_kp = predicted_triangle.get('base_start_kp')
            pred_end_kp = predicted_triangle.get('base_end_kp')

            if pred_start_kp is not None and pred_end_kp is not None:
                hist_start_kps = [loc[0] for loc in hist_locations]
                hist_end_kps = [loc[1] for loc in hist_locations]

                start_kp_range = (min(hist_start_kps) - 5, max(hist_start_kps) + 5)  # ±5km容忍度
                end_kp_range = (min(hist_end_kps) - 5, max(hist_end_kps) + 5)

                if not (start_kp_range[0] <= pred_start_kp <= start_kp_range[1]):
                    warnings.append(f"预测起始位置超出历史范围")

                if not (end_kp_range[0] <= pred_end_kp <= end_kp_range[1]):
                    warnings.append(f"预测结束位置超出历史范围")

        except Exception as e:
            errors.append(f"历史一致性验证失败: {e}")

        return ValidationReport(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def validate_logic_consistency(prediction: Dict) -> ValidationReport:
        """验证逻辑合理性"""
        errors = []
        warnings = []

        try:
            triangle = prediction.get('predicted_triangle', {})

            # 检查时间逻辑
            base_start_time = triangle.get('base_start_time', '')
            apex_time = triangle.get('apex_time', '')
            base_end_time = triangle.get('base_end_time', '')

            if base_start_time and apex_time and base_end_time:
                start_minutes = ValidationUtils._time_to_minutes(base_start_time)
                apex_minutes = ValidationUtils._time_to_minutes(apex_time)
                end_minutes = ValidationUtils._time_to_minutes(base_end_time)

                if start_minutes >= apex_minutes:
                    errors.append("峰值时间必须晚于开始时间")

                if apex_minutes >= end_minutes:
                    errors.append("结束时间必须晚于峰值时间")

                # 检查持续时间合理性
                total_duration = end_minutes - start_minutes
                if total_duration > 600:  # 超过10小时
                    warnings.append("预测拥堵持续时间超过10小时，可能不合理")

            # 检查位置逻辑
            base_start_kp = triangle.get('base_start_kp')
            apex_kp = triangle.get('apex_kp')
            base_end_kp = triangle.get('base_end_kp')

            if all(x is not None for x in [base_start_kp, apex_kp, base_end_kp]):
                if not (base_start_kp <= apex_kp <= base_end_kp):
                    errors.append("顶点位置必须在底边范围内")

                # 检查拥堵范围合理性
                congestion_length = base_end_kp - base_start_kp
                if congestion_length > 100:  # 超过100km
                    warnings.append("预测拥堵范围超过100km，可能不合理")

        except Exception as e:
            errors.append(f"逻辑一致性验证失败: {e}")

        return ValidationReport(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def calculate_final_confidence(prediction: Dict,
                                 physics_validation: ValidationReport,
                                 consistency_validation: ValidationReport,
                                 logic_validation: ValidationReport) -> float:
        """计算最终置信度"""
        base_confidence = prediction.get('predicted_triangle', {}).get('confidence', 0.0)

        # 根据验证结果调整置信度
        confidence_penalty = 0.0

        # 物理约束违反的惩罚
        if not physics_validation.passed:
            confidence_penalty += 0.3

        # 历史一致性违反的惩罚
        if not consistency_validation.passed:
            confidence_penalty += 0.2

        # 逻辑一致性违反的惩罚
        if not logic_validation.passed:
            confidence_penalty += 0.3

        # 警告的惩罚（较轻）
        warning_count = (len(physics_validation.warnings) +
                        len(consistency_validation.warnings) +
                        len(logic_validation.warnings))
        confidence_penalty += min(0.2, warning_count * 0.05)

        final_confidence = max(0.0, base_confidence - confidence_penalty)
        return final_confidence

    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        """将时间字符串转换为分钟"""
        try:
            time_obj = datetime.strptime(time_str, '%H:%M').time()
            return time_obj.hour * 60 + time_obj.minute
        except (ValueError, TypeError):
            return None