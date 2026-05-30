"""
Prediction Expert Agent with LLM integration for traffic congestion prediction.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

from typing import Dict, List, Any
import pandas as pd
from PIL import Image

from .base_agent import BaseAgent, monitor_performance
from ..core.data_structures import Triangle, SystemConfig
from ..core.exceptions import LLMError, DataValidationError
from ..llm.client import LLMClient, MockLLMClient
from ..llm.prompts import PromptTemplates
from ..utils.visualization import TriangleVisualizer


class PredictionExpertAgent(BaseAgent):
    """预测专家Agent - 核心LLM预测逻辑"""

    def __init__(self, config: SystemConfig, use_mock: bool = False):
        super().__init__(config, "PredictionExpertAgent")

        # 初始化LLM客户端
        if use_mock:
            self.llm_client = MockLLMClient()
        else:
            self.llm_client = LLMClient(config.llm_config)

        # 初始化可视化工具
        self.visualizer = TriangleVisualizer()

        # 加载提示词模板
        self.prompt_template = PromptTemplates()

    def validate_input(self, input_data: Dict) -> bool:
        """验证输入数据"""
        required_fields = ['raw_data', 'triangles', 'visualization_data', 'statistics']

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"缺少必要字段: {field}")
                return False

        # 验证数据类型
        if not isinstance(input_data['raw_data'], pd.DataFrame):
            self.logger.error("raw_data必须是DataFrame类型")
            return False

        if not isinstance(input_data['triangles'], list):
            self.logger.error("triangles必须是列表类型")
            return False

        if len(input_data['triangles']) == 0:
            self.logger.error("triangles列表不能为空")
            return False

        return True

    @monitor_performance
    def _process_internal(self, input_data: Dict) -> Dict:
        """
        基于三种数据进行LLM预测

        Args:
            input_data: {
                'raw_data': pd.DataFrame,
                'triangles': List[Triangle],
                'visualization_data': Dict,
                'statistics': Dict
            }
        """
        try:
            # 1. 生成三角形可视化图像
            visualization_image = self._generate_triangle_visualization(
                input_data['triangles'],
                input_data['visualization_data']
            )

            # 2. 准备LLM输入数据
            llm_input = self._prepare_llm_input(
                raw_data=input_data['raw_data'],
                triangles=input_data['triangles'],
                statistics=input_data['statistics']
            )

            # 3. 调用LLM进行预测
            prediction_response = self._call_llm_with_vision(
                prompt=llm_input['prompt'],
                image=visualization_image,
                context=llm_input['context']
            )

            # 4. 解析LLM响应
            parsed_prediction = self._parse_llm_response(prediction_response)

            # 5. 验证预测结果
            self._validate_prediction(parsed_prediction, input_data['triangles'])

            # 6. 添加额外的预测信息
            enhanced_prediction = self._enhance_prediction_result(
                parsed_prediction,
                input_data
            )

            return enhanced_prediction

        except Exception as e:
            self.logger.error(f"预测生成失败: {e}")
            raise

    @monitor_performance
    def _generate_triangle_visualization(self, triangles: List[Triangle], viz_data: Dict) -> Image.Image:
        """生成三角形可视化图像"""
        try:
            # 配置可视化参数
            viz_config = {
                'show_labels': True,
                'highlight_overlaps': True,
                'color_by_year': True
            }

            # 生成图像
            image = self.visualizer.generate_triangle_visualization(triangles, viz_config)

            self.logger.info(f"成功生成可视化图像，三角形数量: {len(triangles)}")
            return image

        except Exception as e:
            self.logger.error(f"可视化生成失败: {e}")
            raise DataValidationError(f"无法生成三角形可视化: {e}")

    @monitor_performance
    def _prepare_llm_input(self, raw_data: pd.DataFrame, triangles: List[Triangle], statistics: Dict) -> Dict:
        """准备LLM输入数据"""
        try:
            # 使用提示词模板生成prompt
            prompt = self.prompt_template.prediction_prompt(raw_data, triangles, statistics)

            # 准备上下文信息
            context = {
                'triangle_count': len(triangles),
                'date_range': statistics.get('date_range'),
                'road_name': statistics.get('road_name'),
                'directions': statistics.get('directions', []),
                'avg_severity': statistics.get('avg_severity', 0.5)
            }

            return {
                'prompt': prompt,
                'context': context
            }

        except Exception as e:
            raise DataValidationError(f"LLM输入准备失败: {e}")

    @monitor_performance
    def _call_llm_with_vision(self, prompt: str, image: Image.Image, context: Dict) -> str:
        """调用支持视觉的LLM"""
        try:
            # 测试LLM连接
            if not self.llm_client.test_connection():
                self.logger.warning("LLM连接测试失败，但继续尝试调用")

            # 调用LLM
            response = self.llm_client.call_with_vision(
                prompt=prompt,
                image=image,
                max_tokens=self.config.llm_config.max_tokens,
                temperature=self.config.llm_config.temperature
            )

            # 验证响应
            if not response or response.strip() == "":
                raise LLMError("LLM返回空响应")

            # 检查响应是否包含预测信息
            prediction_indicators = ["predicted_triangle", "prediction", "预测", "三角形", "拥堵"]
            if not any(indicator in response.lower() for indicator in prediction_indicators):
                self.logger.warning(f"LLM响应可能不包含预测信息: {response[:200]}...")
                # 不抛出异常，因为可能是格式问题

            self.logger.info(f"LLM预测调用成功，响应长度: {len(response)}")
            return response

        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise LLMError(f"LLM预测调用失败: {e}")

    @monitor_performance
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            # 使用LLM客户端的解析功能
            parsed_data = self.llm_client.parse_prediction_response(response)

            # 确保包含必要的字段
            required_fields = ['predicted_triangle', 'explanation']
            for field in required_fields:
                if field not in parsed_data:
                    self.logger.warning(f"LLM响应缺少字段 {field}，使用默认值")
                    if field == 'predicted_triangle':
                        parsed_data[field] = {}
                    elif field == 'explanation':
                        parsed_data[field] = "LLM未提供详细解释"

            # 补充缺失的三角形字段
            triangle = parsed_data['predicted_triangle']
            default_triangle = {
                'apex_time': '18:00',
                'apex_kp': 30.0,
                'base_start_time': '17:30',
                'base_end_time': '18:30',
                'base_start_kp': 25.0,
                'base_end_kp': 35.0,
                'severity': 'medium',
                'confidence': 0.5
            }

            for key, default_value in default_triangle.items():
                if key not in triangle:
                    triangle[key] = default_value
                    self.logger.warning(f"预测三角形缺少字段 {key}，使用默认值: {default_value}")

            return parsed_data

        except Exception as e:
            self.logger.error(f"LLM响应解析失败: {e}")
            raise LLMError(f"预测结果解析失败: {e}")

    def _validate_prediction(self, prediction: Dict, historical_triangles: List[Triangle]) -> None:
        """验证预测结果的基本合理性"""
        try:
            triangle = prediction.get('predicted_triangle', {})

            # 验证必要字段存在
            required_fields = ['apex_time', 'apex_kp', 'base_start_time', 'base_end_time',
                             'base_start_kp', 'base_end_kp', 'severity', 'confidence']

            missing_fields = [field for field in required_fields if field not in triangle]
            if missing_fields:
                raise DataValidationError(f"预测结果缺少必要字段: {missing_fields}")

            # 验证数值范围
            confidence = triangle.get('confidence', 0)
            if not (0 <= confidence <= 1):
                raise DataValidationError(f"置信度必须在0-1之间，当前值: {confidence}")

            # 验证严重程度
            severity = triangle.get('severity', '')
            if severity not in ['high', 'medium', 'low']:
                raise DataValidationError(f"严重程度必须是high/medium/low之一，当前值: {severity}")

            # 验证时间逻辑
            from datetime import datetime
            try:
                apex_time = datetime.strptime(triangle['apex_time'], '%H:%M')
                start_time = datetime.strptime(triangle['base_start_time'], '%H:%M')
                end_time = datetime.strptime(triangle['base_end_time'], '%H:%M')

                if not (start_time <= apex_time <= end_time):
                    raise DataValidationError("峰值时间必须在开始和结束时间之间")

            except ValueError as e:
                raise DataValidationError(f"时间格式错误: {e}")

            # 验证位置逻辑
            apex_kp = float(triangle['apex_kp'])
            start_kp = float(triangle['base_start_kp'])
            end_kp = float(triangle['base_end_kp'])

            if not (start_kp <= apex_kp <= end_kp):
                raise DataValidationError("峰值位置必须在起始和结束位置之间")

            self.logger.info("预测结果验证通过")

        except Exception as e:
            self.logger.error(f"预测验证失败: {e}")
            raise

    def _enhance_prediction_result(self, prediction: Dict, input_data: Dict) -> Dict:
        """增强预测结果，添加额外信息"""
        try:
            # 添加预测元数据
            enhanced_prediction = prediction.copy()

            enhanced_prediction['metadata'] = {
                'prediction_timestamp': pd.Timestamp.now().isoformat(),
                'historical_data_count': len(input_data['triangles']),
                'road_name': input_data['statistics'].get('road_name', ''),
                'directions_analyzed': input_data['statistics'].get('directions', []),
                'llm_model': self.config.llm_config.model
            }

            # 添加预测区间信息
            triangle = enhanced_prediction['predicted_triangle']
            enhanced_prediction['prediction_details'] = {
                'time_range': f"{triangle['base_start_time']}-{triangle['base_end_time']}",
                'location_range': f"KP{triangle['base_start_kp']}-{triangle['base_end_kp']}",
                'peak_time': triangle['apex_time'],
                'peak_location': triangle['apex_kp'],
                'severity': triangle['severity'],
                'duration_minutes': self._calculate_duration(triangle['base_start_time'], triangle['base_end_time']),
                'affected_distance_km': triangle['base_end_kp'] - triangle['base_start_kp']
            }

            return enhanced_prediction

        except Exception as e:
            self.logger.warning(f"预测结果增强失败: {e}")
            return prediction

    def _calculate_duration(self, start_time: str, end_time: str) -> int:
        """计算持续时间（分钟）"""
        try:
            from datetime import datetime
            start = datetime.strptime(start_time, '%H:%M')
            end = datetime.strptime(end_time, '%H:%M')
            duration = (end - start).total_seconds() / 60
            return int(duration)
        except:
            return 120  # 默认2小时