"""
Unit tests for all agents in the multi-agent traffic prediction system.
"""

import unittest
import pandas as pd
from datetime import date, time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from ..core.data_structures import SystemConfig, TriangleConfig, LLMConfig, TrafficEvent, Triangle
from ..core.config import ConfigManager
from ..agents.data_extractor import DataExtractionAgent
from ..agents.prediction_expert import PredictionExpertAgent
from ..agents.validator import ValidationAgent
from ..agents.orchestrator import OrchestratorAgent
from ..utils.data_loader import load_sample_data


class TestDataExtractionAgent(unittest.TestCase):
    """测试数据提取Agent"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            data_path=self.temp_dir,
            output_path=self.temp_dir,
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        self.agent = DataExtractionAgent(self.config)
        self.sample_data = self._create_sample_data()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_data(self) -> pd.DataFrame:
        """创建示例数据"""
        return load_sample_data()

    def _create_sample_file(self, filename: str) -> None:
        """创建示例CSV文件"""
        file_path = os.path.join(self.temp_dir, filename)
        self.sample_data.to_csv(file_path, index=False)

    def test_validate_input_success(self):
        """测试输入验证成功"""
        valid_input = {
            'target_date': '2024-05-15',
            'road_name': '東北道'
        }
        self.assertTrue(self.agent.validate_input(valid_input))

    def test_validate_input_missing_fields(self):
        """测试输入验证失败 - 缺少字段"""
        invalid_input = {'target_date': '2024-05-15'}
        self.assertFalse(self.agent.validate_input(invalid_input))

    def test_validate_input_invalid_date(self):
        """测试输入验证失败 - 无效日期"""
        invalid_input = {
            'target_date': 'invalid-date',
            'road_name': '東北道'
        }
        self.assertFalse(self.agent.validate_input(invalid_input))

    def test_create_traffic_event(self):
        """测试创建TrafficEvent"""
        row = self.sample_data.iloc[0]
        event = self.agent._create_traffic_event(row)

        self.assertIsInstance(event, TrafficEvent)
        self.assertEqual(event.road_name, '東北道')
        self.assertEqual(event.direction, '上')

    def test_create_triangle_from_event(self):
        """测试从事件创建三角形"""
        row = self.sample_data.iloc[0]
        event = self.agent._create_traffic_event(row)
        triangle = self.agent._create_triangle_from_event(event, "test_triangle")

        self.assertIsInstance(triangle, Triangle)
        self.assertEqual(triangle.id, "test_triangle")
        self.assertEqual(len(triangle.vertices), 3)
        self.assertGreater(triangle.area, 0)

    def test_convert_to_triangles(self):
        """测试转换为三角形数据"""
        triangles = self.agent._convert_to_triangles(self.sample_data)

        self.assertIsInstance(triangles, list)
        self.assertEqual(len(triangles), len(self.sample_data))
        for triangle in triangles:
            self.assertIsInstance(triangle, Triangle)

    @patch('os.path.exists')
    def test_load_historical_data_file_not_found(self, mock_exists):
        """测试文件不存在的情况"""
        mock_exists.return_value = False

        with self.assertRaises(Exception):
            self.agent._load_historical_data('2024-05-15', '東北道', '上')

    def test_load_historical_data_success(self):
        """测试成功加载历史数据"""
        # 创建测试文件
        self._create_sample_file('東北道_上_2021_05-15.csv')
        self._create_sample_file('東北道_上_2022_05-15.csv')
        self._create_sample_file('東北道_上_2023_05-15.csv')

        result = self.agent._load_historical_data('2024-05-15', '東北道', '上')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)


class TestPredictionExpertAgent(unittest.TestCase):
    """测试预测专家Agent"""

    def setUp(self):
        """设置测试环境"""
        self.config = SystemConfig(
            data_path="./test_data",
            output_path="./test_output",
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        # 使用模拟LLM
        self.agent = PredictionExpertAgent(self.config, use_mock=True)
        self.sample_triangles = self._create_sample_triangles()

    def _create_sample_triangles(self) -> list:
        """创建示例三角形数据"""
        sample_data = load_sample_data()
        data_agent = DataExtractionAgent(self.config)
        return data_agent._convert_to_triangles(sample_data)

    def test_validate_input_success(self):
        """测试输入验证成功"""
        valid_input = {
            'raw_data': load_sample_data(),
            'triangles': self.sample_triangles,
            'visualization_data': {'triangle_count': 3},
            'statistics': {'total_events': 3}
        }
        self.assertTrue(self.agent.validate_input(valid_input))

    def test_validate_input_missing_fields(self):
        """测试输入验证失败"""
        invalid_input = {
            'raw_data': load_sample_data(),
            'triangles': self.sample_triangles
        }
        self.assertFalse(self.agent.validate_input(invalid_input))

    def test_generate_triangle_visualization(self):
        """测试三角形可视化生成"""
        viz_data = {'triangle_count': len(self.sample_triangles)}
        image = self.agent._generate_triangle_visualization(self.sample_triangles, viz_data)

        self.assertIsNotNone(image)
        # 验证图像是PIL Image对象
        from PIL import Image
        self.assertIsInstance(image, Image.Image)

    def test_prepare_llm_input(self):
        """测试LLM输入准备"""
        raw_data = load_sample_data()
        statistics = {'road_name': '東北道', 'avg_duration': 180}

        llm_input = self.agent._prepare_llm_input(raw_data, self.sample_triangles, statistics)

        self.assertIn('prompt', llm_input)
        self.assertIn('context', llm_input)
        self.assertIsInstance(llm_input['prompt'], str)
        self.assertIsInstance(llm_input['context'], dict)

    def test_parse_llm_response(self):
        """测试LLM响应解析"""
        mock_response = """
        基于图像分析，预测结果如下：
        ```json
        {
            "predicted_triangle": {
                "apex_time": "18:15",
                "apex_kp": 30.2,
                "base_start_time": "17:30",
                "base_end_time": "19:00",
                "base_start_kp": 25.0,
                "base_end_kp": 35.0,
                "severity": "high",
                "confidence": 0.82
            },
            "explanation": "基于历史数据分析...",
            "historical_basis": ["2021年数据", "2022年数据"]
        }
        ```
        """

        parsed = self.agent._parse_llm_response(mock_response)

        self.assertIn('predicted_triangle', parsed)
        self.assertIn('explanation', parsed)
        self.assertEqual(parsed['predicted_triangle']['apex_time'], '18:15')


class TestValidationAgent(unittest.TestCase):
    """测试验证Agent"""

    def setUp(self):
        """设置测试环境"""
        self.config = SystemConfig(
            data_path="./test_data",
            output_path="./test_output",
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        self.agent = ValidationAgent(self.config)

        # 创建示例预测结果
        self.sample_prediction = {
            'predicted_triangle': {
                'apex_time': '18:15',
                'apex_kp': 30.0,
                'base_start_time': '17:30',
                'base_end_time': '19:00',
                'base_start_kp': 25.0,
                'base_end_kp': 35.0,
                'severity': 'high',
                'confidence': 0.8
            },
            'explanation': 'Test prediction'
        }

        # 创建示例历史数据
        sample_data = load_sample_data()
        data_agent = DataExtractionAgent(self.config)
        triangles = data_agent._convert_to_triangles(sample_data)
        self.sample_historical_data = {
            'triangles': triangles,
            'statistics': {'total_events': len(sample_data)}
        }

    def test_validate_input_success(self):
        """测试输入验证成功"""
        valid_input = {
            'prediction': self.sample_prediction,
            'historical_data': self.sample_historical_data
        }
        self.assertTrue(self.agent.validate_input(valid_input))

    def test_validate_input_missing_fields(self):
        """测试输入验证失败"""
        invalid_input = {
            'prediction': self.sample_prediction
        }
        self.assertFalse(self.agent.validate_input(invalid_input))

    def test_validate_physics_constraints(self):
        """测试物理约束验证"""
        validation_result = self.agent._validate_physics_constraints(self.sample_prediction)

        self.assertIsNotNone(validation_result)
        self.assertTrue(validation_result.passed)
        self.assertIsInstance(validation_result.errors, list)
        self.assertIsInstance(validation_result.warnings, list)

    def test_validate_invalid_physics_constraints(self):
        """测试无效物理约束"""
        invalid_prediction = {
            'predicted_triangle': {
                'apex_time': '19:00',  # 峰值时间晚于结束时间
                'apex_kp': 30.0,
                'base_start_time': '17:30',
                'base_end_time': '18:30',  # 结束时间早于峰值时间
                'base_start_kp': 25.0,
                'base_end_kp': 35.0,
                'severity': 'high',
                'confidence': 0.8
            }
        }

        validation_result = self.agent._validate_physics_constraints(invalid_prediction)
        self.assertFalse(validation_result.passed)
        self.assertGreater(len(validation_result.errors), 0)

    def test_calculate_final_confidence(self):
        """测试最终置信度计算"""
        from ..core.data_structures import ValidationReport

        physics_validation = ValidationReport(passed=True)
        consistency_validation = ValidationReport(passed=True)
        logic_validation = ValidationReport(passed=True)

        final_confidence = self.agent._calculate_final_confidence(
            self.sample_prediction,
            physics_validation,
            consistency_validation,
            logic_validation
        )

        self.assertGreaterEqual(final_confidence, 0.0)
        self.assertLessEqual(final_confidence, 1.0)


class TestOrchestratorAgent(unittest.TestCase):
    """测试协调器Agent"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            data_path=self.temp_dir,
            output_path=self.temp_dir,
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        # 使用模拟LLM
        self.agent = OrchestratorAgent(self.config, use_mock_llm=True)
        self._create_test_data()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data(self):
        """创建测试数据文件"""
        sample_data = load_sample_data()
        # 创建历史数据文件
        for year in [2021, 2022, 2023]:
            filename = f'東北道_上_{year}_05-15.csv'
            file_path = os.path.join(self.temp_dir, filename)
            sample_data.to_csv(file_path, index=False)

    def test_validate_input_success(self):
        """测试输入验证成功"""
        valid_input = {
            'target_date': '2024-05-15',
            'road_name': '東北道',
            'direction': '上'
        }
        self.assertTrue(self.agent.validate_input(valid_input))

    def test_validate_input_missing_fields(self):
        """测试输入验证失败"""
        invalid_input = {
            'target_date': '2024-05-15'
        }
        self.assertFalse(self.agent.validate_input(invalid_input))

    def test_get_agents_status(self):
        """测试获取Agent状态"""
        status = self.agent.get_agents_status()

        self.assertIsInstance(status, dict)
        self.assertIn('data_extractor', status)
        self.assertIn('prediction_expert', status)
        self.assertIn('validator', status)

    def test_reset_all_agents(self):
        """测试重置所有Agent"""
        self.agent.reset_all_agents()

        # 验证执行日志被清空
        self.assertEqual(len(self.agent.execution_log), 0)


if __name__ == '__main__':
    unittest.main()