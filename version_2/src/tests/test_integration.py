"""
Integration tests for the multi-agent traffic prediction system.
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch

from ..traffic_prediction_system import TrafficPredictionSystem, create_system
from ..core.data_structures import SystemConfig, TriangleConfig, LLMConfig
from ..utils.data_loader import load_sample_data


class TestTrafficPredictionSystemIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            data_path=self.temp_dir,
            output_path=self.temp_dir,
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        self._create_test_data()

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data(self):
        """创建测试数据文件"""
        sample_data = load_sample_data()
        # 创建历史数据文件
        for year in [2021, 2022, 2023]:
            filename = f'東北道_上_{year}_05-15.csv'
            file_path = os.path.join(self.temp_dir, filename)
            sample_data.to_csv(file_path, index=False)

    def test_system_initialization(self):
        """测试系统初始化"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.orchestrator)

    def test_system_status(self):
        """测试系统状态检查"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)
        status = system.get_system_status()

        self.assertIsInstance(status, dict)
        self.assertIn('system_status', status)
        self.assertIn('orchestrator', status)
        self.assertIn('agents', status)

    def test_configuration_validation(self):
        """测试配置验证"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)
        validation_result = system.validate_configuration()

        self.assertIsInstance(validation_result, dict)
        self.assertIn('valid', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('warnings', validation_result)

    def test_full_prediction_workflow(self):
        """测试完整预测流程"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        request = {
            'target_date': '2024-05-15',
            'road_name': '東北道',
            'direction': '上'
        }

        # 执行预测
        result = system.predict(request)

        # 验证结果结构
        self.assertIsInstance(result, dict)
        self.assertIn('request', result)
        self.assertIn('prediction', result)
        self.assertIn('validation', result)
        self.assertIn('metadata', result)
        self.assertIn('summary', result)

        # 验证预测内容
        prediction = result['prediction']
        self.assertIn('predicted_triangle', prediction)
        self.assertIn('explanation', prediction)

        # 验证验证结果
        validation = result['validation']
        self.assertIn('final_confidence', validation)
        self.assertIn('validation_passed', validation)

        # 验证置信度范围
        confidence = validation['final_confidence']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_quick_predict_interface(self):
        """测试快速预测接口"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        result = system.quick_predict('2024-05-15', '東北道', '上')

        self.assertIsInstance(result, dict)
        self.assertIn('prediction_summary', result)
        self.assertIn('confidence', result)
        self.assertIn('validation_passed', result)
        self.assertIn('recommendation', result)

    def test_system_reset(self):
        """测试系统重置"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        # 执行一次预测
        system.quick_predict('2024-05-15', '東北道', '上')

        # 重置系统
        system.reset_system()

        # 验证系统状态
        status = system.get_system_status()
        self.assertEqual(status['system_status'], 'healthy')

    def test_create_system_convenience_function(self):
        """测试便捷创建函数"""
        # 使用配置路径创建
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data_path: ./test_data
output_path: ./test_output
llm_config:
  model: gpt-4-vision-preview
  api_key: test-key
""")
            config_path = f.name

        try:
            system = create_system(config_path=config_path, use_mock_llm=True)
            self.assertIsNotNone(system)
        finally:
            os.unlink(config_path)

    def test_error_handling(self):
        """测试错误处理"""
        # 创建无效配置的系统
        invalid_config = SystemConfig(
            data_path="/nonexistent/path",
            output_path=self.temp_dir,
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )

        system = TrafficPredictionSystem(config=invalid_config, use_mock_llm=True)

        # 尝试执行预测，应该失败
        with self.assertRaises(Exception):
            system.predict({
                'target_date': '2024-05-15',
                'road_name': '不存在的道路'
            })

    def test_prediction_with_different_parameters(self):
        """测试不同参数的预测"""
        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        # 测试不同的参数组合
        test_cases = [
            {
                'target_date': '2024-05-15',
                'road_name': '東北道',
                'direction': '上'
            },
            {
                'target_date': '2024-05-15',
                'road_name': '東北道',
                'direction': '下'
            },
            {
                'target_date': '2024-05-15',
                'road_name': '東北道',
                'time_range': '17:00-19:00'
            }
        ]

        for request in test_cases:
            with self.subTest(request=request):
                try:
                    result = system.predict(request)
                    self.assertIsInstance(result, dict)
                    self.assertIn('prediction', result)
                except Exception as e:
                    # 某些情况下可能因为数据不足而失败，这是可以接受的
                    self.assertIn('数据', str(e).lower())


class TestSystemPerformance(unittest.TestCase):
    """性能测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            data_path=self.temp_dir,
            output_path=self.temp_dir,
            triangle_config=TriangleConfig(),
            llm_config=LLMConfig(api_key="test-key")
        )
        self._create_test_data()

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data(self):
        """创建测试数据文件"""
        sample_data = load_sample_data()
        # 创建多个历史数据文件
        for year in [2021, 2022, 2023]:
            for month in ['05', '06', '07']:
                for day in ['15', '16']:
                    filename = f'東北道_上_{year}_{month}-{day}.csv'
                    file_path = os.path.join(self.temp_dir, filename)
                    sample_data.to_csv(file_path, index=False)

    def test_prediction_time_limit(self):
        """测试预测时间限制"""
        import time

        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        start_time = time.time()
        result = system.quick_predict('2024-05-15', '東北道', '上')
        execution_time = time.time() - start_time

        # 验证执行时间在合理范围内（应该 < 30秒，但在测试环境中应该更快）
        self.assertLess(execution_time, 10.0)
        self.assertIsInstance(result, dict)

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        system = TrafficPredictionSystem(config=self.config, use_mock_llm=True)

        # 执行多次预测
        for i in range(5):
            system.quick_predict('2024-05-15', '東北道', '上')

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证内存增长在合理范围内（< 100MB）
        self.assertLess(memory_increase, 100)


if __name__ == '__main__':
    unittest.main()