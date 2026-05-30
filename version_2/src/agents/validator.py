"""
Validation Agent for prediction result verification.
Based on AI_DEVELOPMENT_GUIDE.md specifications.
"""

from typing import Dict, List, Any
from datetime import datetime

from .base_agent import BaseAgent, monitor_performance
from ..core.data_structures import Triangle, ValidationReport, SystemConfig
from ..core.exceptions import PredictionValidationError
from ..utils.validation import ValidationUtils


class ValidationAgent(BaseAgent):
    """验证Agent - 负责预测结果的验证和质量控制"""

    def __init__(self, config: SystemConfig):
        super().__init__(config, "ValidationAgent")
        self.validation_utils = ValidationUtils()

    def validate_input(self, input_data: Dict) -> bool:
        """验证输入数据"""
        required_fields = ['prediction', 'historical_data']

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"缺少必要字段: {field}")
                return False

        # 验证预测数据结构
        prediction = input_data['prediction']
        if 'predicted_triangle' not in prediction:
            self.logger.error("预测数据缺少predicted_triangle字段")
            return False

        return True

    @monitor_performance
    def _process_internal(self, input_data: Dict) -> Dict:
        """
        验证预测结果的合理性

        Args:
            input_data: {
                'prediction': Dict,      # 预测结果
                'historical_data': Dict  # 历史数据
            }
        """
        try:
            prediction = input_data['prediction']
            historical_data = input_data['historical_data']

            # 1. 物理约束验证
            physics_validation = self._validate_physics_constraints(prediction)

            # 2. 历史一致性验证
            consistency_validation = self._validate_historical_consistency(
                prediction, historical_data
            )

            # 3. 逻辑合理性验证
            logic_validation = self._validate_logic_consistency(prediction)

            # 4. 计算综合置信度
            final_confidence = self._calculate_final_confidence(
                prediction,
                physics_validation,
                consistency_validation,
                logic_validation
            )

            # 5. 生成验证报告
            validation_report = self._generate_validation_report(
                physics_validation,
                consistency_validation,
                logic_validation,
                final_confidence
            )

            # 6. 检查是否需要拒绝预测
            critical_failures = self._check_critical_failures(
                physics_validation,
                consistency_validation,
                logic_validation
            )

            if critical_failures:
                self.logger.error(f"预测验证发现严重问题: {critical_failures}")
                # 根据严格的错误处理原则，明确报告问题
                raise PredictionValidationError(f"预测验证失败: {'; '.join(critical_failures)}")

            # 7. 更新预测结果的置信度
            validated_prediction = prediction.copy()
            validated_prediction['predicted_triangle']['confidence'] = final_confidence

            return {
                'validated_prediction': validated_prediction,
                'final_confidence': final_confidence,
                'validation_report': validation_report,
                'validation_details': {
                    'physics': physics_validation.__dict__,
                    'consistency': consistency_validation.__dict__,
                    'logic': logic_validation.__dict__
                },
                'validation_passed': not critical_failures
            }

        except Exception as e:
            self.logger.error(f"验证过程失败: {e}")
            raise

    @monitor_performance
    def _validate_physics_constraints(self, prediction: Dict) -> ValidationReport:
        """验证物理约束"""
        return self.validation_utils.validate_physics_constraints(prediction)

    @monitor_performance
    def _validate_historical_consistency(self, prediction: Dict, historical_data: Dict) -> ValidationReport:
        """验证历史一致性"""
        return self.validation_utils.validate_historical_consistency(prediction, historical_data)

    @monitor_performance
    def _validate_logic_consistency(self, prediction: Dict) -> ValidationReport:
        """验证逻辑合理性"""
        return self.validation_utils.validate_logic_consistency(prediction)

    def _calculate_final_confidence(self,
                                  prediction: Dict,
                                  physics_validation: ValidationReport,
                                  consistency_validation: ValidationReport,
                                  logic_validation: ValidationReport) -> float:
        """计算最终置信度"""
        return self.validation_utils.calculate_final_confidence(
            prediction, physics_validation, consistency_validation, logic_validation
        )

    def _check_critical_failures(self,
                                physics_validation: ValidationReport,
                                consistency_validation: ValidationReport,
                                logic_validation: ValidationReport) -> List[str]:
        """检查严重的验证失败"""
        critical_failures = []

        # 物理约束必须通过
        if not physics_validation.passed:
            critical_failures.extend([f"物理约束: {error}" for error in physics_validation.errors])

        # 逻辑一致性必须通过
        if not logic_validation.passed:
            critical_failures.extend([f"逻辑一致性: {error}" for error in logic_validation.errors])

        # 历史一致性失败是警告，不是严重错误，但要记录
        if not consistency_validation.passed:
            self.logger.warning(f"历史一致性验证失败: {consistency_validation.errors}")

        return critical_failures

    @monitor_performance
    def _generate_validation_report(self,
                                  physics_validation: ValidationReport,
                                  consistency_validation: ValidationReport,
                                  logic_validation: ValidationReport,
                                  final_confidence: float) -> str:
        """生成验证报告"""
        report_lines = ["## 预测验证报告\n"]

        # 总体评估
        overall_status = "通过" if final_confidence > 0.3 else "不通过"
        report_lines.append(f"**总体评估**: {overall_status}")
        report_lines.append(f"**最终置信度**: {final_confidence:.2f}\n")

        # 物理约束验证
        report_lines.append("### 1. 物理约束验证")
        if physics_validation.passed:
            report_lines.append("✅ **通过** - 预测符合物理约束")
        else:
            report_lines.append("❌ **失败** - 预测违反物理约束")
            for error in physics_validation.errors:
                report_lines.append(f"   - {error}")

        if physics_validation.warnings:
            report_lines.append("⚠️ **警告**:")
            for warning in physics_validation.warnings:
                report_lines.append(f"   - {warning}")

        # 历史一致性验证
        report_lines.append("\n### 2. 历史一致性验证")
        if consistency_validation.passed:
            report_lines.append("✅ **通过** - 预测与历史数据一致")
        else:
            report_lines.append("⚠️ **需注意** - 预测与历史数据存在差异")
            for error in consistency_validation.errors:
                report_lines.append(f"   - {error}")

        if consistency_validation.warnings:
            for warning in consistency_validation.warnings:
                report_lines.append(f"   - {warning}")

        # 逻辑一致性验证
        report_lines.append("\n### 3. 逻辑一致性验证")
        if logic_validation.passed:
            report_lines.append("✅ **通过** - 预测逻辑一致")
        else:
            report_lines.append("❌ **失败** - 预测存在逻辑问题")
            for error in logic_validation.errors:
                report_lines.append(f"   - {error}")

        if logic_validation.warnings:
            report_lines.append("⚠️ **警告**:")
            for warning in logic_validation.warnings:
                report_lines.append(f"   - {warning}")

        # 建议
        report_lines.append("\n### 4. 建议")
        suggestions = self._generate_suggestions(
            physics_validation, consistency_validation, logic_validation, final_confidence
        )
        for suggestion in suggestions:
            report_lines.append(f"- {suggestion}")

        return "\n".join(report_lines)

    def _generate_suggestions(self,
                            physics_validation: ValidationReport,
                            consistency_validation: ValidationReport,
                            logic_validation: ValidationReport,
                            final_confidence: float) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if final_confidence < 0.5:
            suggestions.append("置信度较低，建议谨慎使用此预测结果")

        if not physics_validation.passed:
            suggestions.append("需要修正预测的物理约束问题")

        if not logic_validation.passed:
            suggestions.append("需要修正预测的逻辑一致性问题")

        if not consistency_validation.passed:
            suggestions.append("预测结果与历史模式存在较大差异，建议进一步分析")

        if physics_validation.warnings or consistency_validation.warnings or logic_validation.warnings:
            suggestions.append("注意警告信息，考虑调整预测参数")

        if final_confidence > 0.8:
            suggestions.append("预测质量较高，可以作为决策参考")

        if not suggestions:
            suggestions.append("预测结果整体合理，可以使用")

        return suggestions

    def get_validation_summary(self, validation_result: Dict) -> Dict:
        """获取验证摘要"""
        return {
            'validation_passed': validation_result.get('validation_passed', False),
            'final_confidence': validation_result.get('final_confidence', 0.0),
            'physics_passed': validation_result.get('validation_details', {}).get('physics', {}).get('passed', False),
            'consistency_passed': validation_result.get('validation_details', {}).get('consistency', {}).get('passed', False),
            'logic_passed': validation_result.get('validation_details', {}).get('logic', {}).get('passed', False),
            'total_errors': len(validation_result.get('validation_details', {}).get('physics', {}).get('errors', [])) +
                           len(validation_result.get('validation_details', {}).get('logic', {}).get('errors', [])),
            'total_warnings': len(validation_result.get('validation_details', {}).get('physics', {}).get('warnings', [])) +
                             len(validation_result.get('validation_details', {}).get('consistency', {}).get('warnings', [])) +
                             len(validation_result.get('validation_details', {}).get('logic', {}).get('warnings', []))
        }