"""
Tests for core data structures and validation.
"""

import unittest
from datetime import date, time
from ..core.data_structures import TrafficEvent, Triangle, PredictionRequest, ValidationReport
from ..core.exceptions import DataValidationError, TriangleValidationError


class TestTrafficEvent(unittest.TestCase):
    """测试TrafficEvent数据结构"""

    def test_valid_traffic_event(self):
        """测试有效的交通事件"""
        event = TrafficEvent(
            date=date(2024, 5, 15),
            direction="上",
            cause="交通集中",
            road_name="東北道",
            start_time=time(17, 30),
            peak_time=time(18, 15),
            peak_length=8.5,
            start_kp=25.4,
            initial_length=3.2,
            duration=180
        )

        # 验证应该通过
        event.validate()

    def test_invalid_peak_time(self):
        """测试无效的峰值时间"""
        event = TrafficEvent(
            date=date(2024, 5, 15),
            direction="上",
            cause="交通集中",
            road_name="東北道",
            start_time=time(18, 30),  # 开始时间晚于峰值时间
            peak_time=time(18, 15),
            peak_length=8.5,
            start_kp=25.4,
            initial_length=3.2,
            duration=180
        )

        with self.assertRaises(DataValidationError):
            event.validate()

    def test_invalid_direction(self):
        """测试无效的方向"""
        event = TrafficEvent(
            date=date(2024, 5, 15),
            direction="中",  # 无效方向
            cause="交通集中",
            road_name="東北道",
            start_time=time(17, 30),
            peak_time=time(18, 15),
            peak_length=8.5,
            start_kp=25.4,
            initial_length=3.2,
            duration=180
        )

        with self.assertRaises(DataValidationError):
            event.validate()

    def test_invalid_peak_length(self):
        """测试无效的峰值长度"""
        event = TrafficEvent(
            date=date(2024, 5, 15),
            direction="上",
            cause="交通集中",
            road_name="東北道",
            start_time=time(17, 30),
            peak_time=time(18, 15),
            peak_length=2.0,  # 峰值长度小于初始长度
            start_kp=25.4,
            initial_length=3.2,
            duration=180
        )

        with self.assertRaises(DataValidationError):
            event.validate()


class TestTriangle(unittest.TestCase):
    """测试Triangle数据结构"""

    def setUp(self):
        """设置测试数据"""
        self.valid_event = TrafficEvent(
            date=date(2024, 5, 15),
            direction="上",
            cause="交通集中",
            road_name="東北道",
            start_time=time(17, 30),
            peak_time=time(18, 15),
            peak_length=8.5,
            start_kp=25.4,
            initial_length=3.2,
            duration=180
        )

    def test_valid_triangle(self):
        """测试有效的三角形"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (30.0, 1095), (35.0, 1050)],
            center=(30.0, 1065),
            area=150.0,
            kp_start=25.0,
            kp_end=35.0,
            time_start=1050,
            time_peak=1095,
            duration=180,
            severity=0.7,
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        # 验证应该通过
        triangle.validate()

    def test_invalid_vertices_count(self):
        """测试无效的顶点数量"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (35.0, 1050)],  # 只有两个顶点
            center=(30.0, 1065),
            area=150.0,
            kp_start=25.0,
            kp_end=35.0,
            time_start=1050,
            time_peak=1095,
            duration=180,
            severity=0.7,
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        with self.assertRaises(TriangleValidationError):
            triangle.validate()

    def test_invalid_area(self):
        """测试无效的面积"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (30.0, 1095), (35.0, 1050)],
            center=(30.0, 1065),
            area=0.0,  # 面积为0
            kp_start=25.0,
            kp_end=35.0,
            time_start=1050,
            time_peak=1095,
            duration=180,
            severity=0.7,
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        with self.assertRaises(TriangleValidationError):
            triangle.validate()

    def test_invalid_kp_range(self):
        """测试无效的KP范围"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (30.0, 1095), (35.0, 1050)],
            center=(30.0, 1065),
            area=150.0,
            kp_start=35.0,  # 起始KP大于结束KP
            kp_end=25.0,
            time_start=1050,
            time_peak=1095,
            duration=180,
            severity=0.7,
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        with self.assertRaises(TriangleValidationError):
            triangle.validate()

    def test_invalid_time_sequence(self):
        """测试无效的时间序列"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (30.0, 1095), (35.0, 1050)],
            center=(30.0, 1065),
            area=150.0,
            kp_start=25.0,
            kp_end=35.0,
            time_start=1095,  # 开始时间晚于峰值时间
            time_peak=1050,
            duration=180,
            severity=0.7,
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        with self.assertRaises(TriangleValidationError):
            triangle.validate()

    def test_invalid_severity(self):
        """测试无效的严重程度"""
        triangle = Triangle(
            id="test_triangle",
            vertices=[(25.0, 1050), (30.0, 1095), (35.0, 1050)],
            center=(30.0, 1065),
            area=150.0,
            kp_start=25.0,
            kp_end=35.0,
            time_start=1050,
            time_peak=1095,
            duration=180,
            severity=1.5,  # 严重程度超出范围
            direction="上",
            road_name="東北道",
            source_event=self.valid_event
        )

        with self.assertRaises(TriangleValidationError):
            triangle.validate()


class TestPredictionRequest(unittest.TestCase):
    """测试PredictionRequest数据结构"""

    def test_valid_request(self):
        """测试有效的预测请求"""
        request = PredictionRequest(
            target_date="2024-05-15",
            road_name="東北道",
            direction="上",
            time_range="17:00-19:00"
        )

        # 验证应该通过
        request.validate()

    def test_invalid_date_format(self):
        """测试无效的日期格式"""
        request = PredictionRequest(
            target_date="05/15/2024",  # 错误的日期格式
            road_name="東北道"
        )

        with self.assertRaises(DataValidationError):
            request.validate()

    def test_invalid_direction(self):
        """测试无效的方向"""
        request = PredictionRequest(
            target_date="2024-05-15",
            road_name="東北道",
            direction="东"  # 错误的方向
        )

        with self.assertRaises(DataValidationError):
            request.validate()

    def test_invalid_time_range(self):
        """测试无效的时间范围"""
        request = PredictionRequest(
            target_date="2024-05-15",
            road_name="東北道",
            time_range="17:00-25:00"  # 无效时间
        )

        with self.assertRaises(DataValidationError):
            request.validate()

    def test_optional_fields(self):
        """测试可选字段"""
        request = PredictionRequest(
            target_date="2024-05-15",
            road_name="東北道"
            # direction和time_range都是可选的
        )

        # 验证应该通过
        request.validate()


class TestValidationReport(unittest.TestCase):
    """测试ValidationReport数据结构"""

    def test_validation_report_creation(self):
        """测试验证报告创建"""
        report = ValidationReport(
            passed=True,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            confidence_adjustments={"physics": -0.1}
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(report.errors), 2)
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("physics", report.confidence_adjustments)

    def test_validation_report_defaults(self):
        """测试验证报告默认值"""
        report = ValidationReport(passed=False)

        self.assertFalse(report.passed)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(len(report.warnings), 0)
        self.assertEqual(len(report.confidence_adjustments), 0)


if __name__ == '__main__':
    unittest.main()