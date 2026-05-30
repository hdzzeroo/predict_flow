"""
Orchestrator Agent for coordinating the multi-agent traffic prediction system.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time

from .base_agent import BaseAgent, monitor_performance
from .data_extractor import DataExtractionAgent
from .prediction_expert import PredictionExpertAgent
from .validator import ValidationAgent
from ..core.data_structures import PredictionRequest, SystemConfig
from ..core.exceptions import AgentError, DataValidationError
from ..llm.prompts import PromptTemplates


class OrchestratorAgent(BaseAgent):
    """总协调Agent - 负责流程控制"""

    def __init__(self, config: SystemConfig, use_mock_llm: bool = False):
        super().__init__(config, "OrchestratorAgent")
        self.agents = self._initialize_agents(use_mock_llm)
        self.execution_log = []

    def _initialize_agents(self, use_mock_llm: bool = False) -> Dict:
        """初始化所有Agent"""
        try:
            agents = {
                'data_extractor': DataExtractionAgent(self.config),
                'prediction_expert': PredictionExpertAgent(self.config, use_mock=use_mock_llm),
                'validator': ValidationAgent(self.config)
            }

            # 验证所有Agent都初始化成功
            for agent_name, agent in agents.items():
                status = agent.get_status()
                if status['status'] != 'initialized':
                    raise AgentError(f"Agent {agent_name} 初始化失败")

            self.logger.info(f"成功初始化 {len(agents)} 个Agent")
            return agents

        except Exception as e:
            self.logger.error(f"Agent初始化失败: {e}")
            raise AgentError(f"Agent初始化失败: {e}")

    def validate_input(self, input_data: Dict) -> bool:
        """验证输入数据"""
        required_fields = ['target_date', 'road_name']
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"缺少必要字段: {field}")
                return False

        # 验证并解析预测请求
        try:
            request = PredictionRequest(
                target_date=input_data['target_date'],
                road_name=input_data['road_name'],
                direction=input_data.get('direction'),
                time_range=input_data.get('time_range')
            )
            request.validate()
            return True
        except Exception as e:
            self.logger.error(f"预测请求验证失败: {e}")
            return False

    @monitor_performance
    def _process_internal(self, input_data: Dict) -> Dict:
        """
        处理预测请求的主流程

        Args:
            input_data: {
                'target_date': '2024-05-15',
                'road_name': '東北道',
                'direction': '上',  # 可选
                'time_range': '17:00-19:00'  # 可选
            }

        Returns:
            完整的预测结果和执行日志
        """
        workflow_start_time = time.time()
        self.execution_log = []

        try:
            # 记录开始
            self._log_step("开始预测流程", {
                'target_date': input_data['target_date'],
                'road_name': input_data['road_name'],
                'direction': input_data.get('direction', '未指定'),
                'time_range': input_data.get('time_range', '未指定')
            })

            # Phase 1: 数据提取
            self.logger.info("Phase 1: 开始数据提取")
            historical_data = self._execute_data_extraction(input_data)

            # Phase 2: 预测生成
            self.logger.info("Phase 2: 开始预测生成")
            prediction = self._execute_prediction(historical_data)

            # Phase 3: 预测验证
            self.logger.info("Phase 3: 开始预测验证")
            validated_result = self._execute_validation(prediction, historical_data)

            # Phase 4: 结果整合
            self.logger.info("Phase 4: 整合最终结果")
            final_result = self._integrate_final_result(
                input_data, historical_data, prediction, validated_result, workflow_start_time
            )

            self._log_step("预测流程完成", {
                'final_confidence': final_result.get('final_confidence', 0),
                'total_duration': time.time() - workflow_start_time
            })

            return final_result

        except Exception as e:
            self._log_step("预测流程失败", {'error': str(e)})
            self.logger.error(f"协调器处理失败: {e}")
            raise AgentError(f"预测流程执行失败: {e}")

    @monitor_performance
    def _execute_data_extraction(self, input_data: Dict) -> Dict:
        """执行数据提取阶段"""
        try:
            extraction_input = {
                'target_date': input_data['target_date'],
                'road_name': input_data['road_name'],
                'direction': input_data.get('direction')
            }

            result = self.agents['data_extractor'].process(extraction_input)

            if not result.get('success', False):
                raise AgentError("数据提取失败")

            historical_data = result['data']
            self._log_step("数据提取完成", {
                'triangles_count': len(historical_data.get('triangles', [])),
                'raw_data_records': len(historical_data.get('raw_data', [])),
                'execution_time': result.get('execution_time', 0)
            })

            return historical_data

        except Exception as e:
            self.logger.error(f"数据提取阶段失败: {e}")
            raise

    @monitor_performance
    def _execute_prediction(self, historical_data: Dict) -> Dict:
        """执行预测生成阶段"""
        try:
            result = self.agents['prediction_expert'].process(historical_data)

            if not result.get('success', False):
                raise AgentError("预测生成失败")

            prediction = result['data']
            self._log_step("预测生成完成", {
                'predicted_triangle': prediction.get('predicted_triangle', {}),
                'confidence': prediction.get('predicted_triangle', {}).get('confidence', 0),
                'execution_time': result.get('execution_time', 0)
            })

            return prediction

        except Exception as e:
            self.logger.error(f"预测生成阶段失败: {e}")
            raise

    @monitor_performance
    def _execute_validation(self, prediction: Dict, historical_data: Dict) -> Dict:
        """执行预测验证阶段"""
        try:
            validation_input = {
                'prediction': prediction,
                'historical_data': historical_data
            }

            result = self.agents['validator'].process(validation_input)

            if not result.get('success', False):
                raise AgentError("预测验证失败")

            validated_result = result['data']
            self._log_step("预测验证完成", {
                'validation_passed': validated_result.get('validation_passed', False),
                'final_confidence': validated_result.get('final_confidence', 0),
                'execution_time': result.get('execution_time', 0)
            })

            return validated_result

        except Exception as e:
            self.logger.error(f"预测验证阶段失败: {e}")
            raise

    def _integrate_final_result(self,
                              input_data: Dict,
                              historical_data: Dict,
                              prediction: Dict,
                              validated_result: Dict,
                              workflow_start_time: float) -> Dict:
        """整合最终结果"""
        try:
            # 基础预测信息
            predicted_triangle = validated_result['validated_prediction']['predicted_triangle']

            # 构建最终结果
            final_result = {
                # 预测请求信息
                'request': {
                    'target_date': input_data['target_date'],
                    'road_name': input_data['road_name'],
                    'direction': input_data.get('direction'),
                    'time_range': input_data.get('time_range')
                },

                # 预测结果
                'prediction': {
                    'predicted_triangle': predicted_triangle,
                    'prediction_details': prediction.get('prediction_details', {}),
                    'explanation': prediction.get('explanation', ''),
                    'historical_basis': prediction.get('historical_basis', [])
                },

                # 验证结果
                'validation': {
                    'validation_passed': validated_result.get('validation_passed', False),
                    'final_confidence': validated_result.get('final_confidence', 0),
                    'validation_report': validated_result.get('validation_report', ''),
                    'validation_summary': self.agents['validator'].get_validation_summary(validated_result)
                },

                # 元数据
                'metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'total_execution_time': time.time() - workflow_start_time,
                    'historical_data_count': len(historical_data.get('triangles', [])),
                    'workflow_version': '2.0',
                    'agents_used': list(self.agents.keys())
                },

                # 执行日志
                'execution_log': self.execution_log.copy(),

                # 简化的摘要信息
                'summary': self._generate_summary(predicted_triangle, validated_result)
            }

            return final_result

        except Exception as e:
            self.logger.error(f"结果整合失败: {e}")
            raise

    def _generate_summary(self, predicted_triangle: Dict, validated_result: Dict) -> Dict:
        """生成预测摘要"""
        return {
            'prediction_date': predicted_triangle.get('apex_time', 'N/A'),
            'prediction_location': f"KP{predicted_triangle.get('apex_kp', 'N/A')}",
            'time_range': f"{predicted_triangle.get('base_start_time', 'N/A')}-{predicted_triangle.get('base_end_time', 'N/A')}",
            'location_range': f"KP{predicted_triangle.get('base_start_kp', 'N/A')}-{predicted_triangle.get('base_end_kp', 'N/A')}",
            'severity': predicted_triangle.get('severity', 'unknown'),
            'confidence': validated_result.get('final_confidence', 0),
            'validation_status': 'PASSED' if validated_result.get('validation_passed', False) else 'FAILED',
            'recommendation': self._get_recommendation(
                predicted_triangle.get('severity', 'unknown'),
                validated_result.get('final_confidence', 0)
            )
        }

    def _get_recommendation(self, severity: str, confidence: float) -> str:
        """根据预测结果给出建议"""
        if confidence < 0.3:
            return "预测置信度较低，建议谨慎使用"
        elif confidence < 0.6:
            return "预测结果需要进一步验证"
        elif severity == 'high' and confidence >= 0.7:
            return "高置信度预测严重拥堵，建议采取预防措施"
        elif severity == 'medium' and confidence >= 0.6:
            return "中等程度拥堵预测，建议关注交通状况"
        else:
            return "预测结果可作为参考"

    def _log_step(self, step_name: str, details: Dict) -> None:
        """记录执行步骤"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'details': details
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"执行步骤: {step_name}", extra=details)

    def get_agents_status(self) -> Dict:
        """获取所有Agent的状态"""
        agents_status = {}
        for agent_name, agent in self.agents.items():
            agents_status[agent_name] = agent.get_status()
        return agents_status

    def reset_all_agents(self) -> None:
        """重置所有Agent状态"""
        for agent in self.agents.values():
            agent.reset_status()
        self.execution_log = []
        self.logger.info("所有Agent状态已重置")

    def quick_predict(self, target_date: str, road_name: str, direction: str = None) -> Dict:
        """快速预测接口"""
        request = {
            'target_date': target_date,
            'road_name': road_name
        }
        if direction:
            request['direction'] = direction

        return self.process(request)