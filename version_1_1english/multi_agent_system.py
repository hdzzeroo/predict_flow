"""
多Agent交通预测系统
模拟人类专家团队进行交通拥堵预测
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from abc import ABC, abstractmethod

class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    OBSERVATION = "observation"
    SUGGESTION = "suggestion"
    VALIDATION = "validation"
    DECISION = "decision"

@dataclass
class Message:
    """Agent间通信消息"""
    sender: str
    receiver: str
    type: MessageType
    content: Dict[str, Any]
    context: Optional[Dict] = None

class BaseAgent(ABC):
    """基础Agent类"""

    def __init__(self, name: str, role: str, llm=None):
        self.name = name
        self.role = role
        self.llm = llm
        self.memory = []  # 记忆历史交互
        self.observations = []  # 记录观察到的模式

    @abstractmethod
    def process(self, message: Message) -> Message:
        """处理消息并返回响应"""
        pass

    def think(self, context: str) -> str:
        """内部思考过程"""
        if self.llm:
            prompt = f"""
            作为{self.role}，基于以下情况：
            {context}

            我的分析和建议是：
            """
            return self.llm.invoke(prompt)
        return f"{self.name}的分析结果"

    def remember(self, information: Any):
        """存储重要信息到记忆"""
        self.memory.append(information)

class DataAnalystAgent(BaseAgent):
    """数据分析Agent - 负责数据处理和特征提取"""

    def __init__(self, llm=None):
        super().__init__("DataAnalyst", "数据分析专家", llm)

    def process(self, message: Message) -> Message:
        """处理数据分析请求"""
        if message.type == MessageType.REQUEST:
            data = message.content.get('data', {})

            # 分析数据模式
            analysis = self.analyze_data_patterns(data)

            # 提取关键特征
            features = self.extract_features(data)

            # 返回分析结果
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content={
                    'analysis': analysis,
                    'features': features,
                    'statistics': self.compute_statistics(data)
                }
            )

    def analyze_data_patterns(self, data: Dict) -> Dict:
        """分析数据中的模式"""
        thought = self.think(f"""
        分析以下交通数据的模式：
        - 数据时间范围：{data.get('time_range')}
        - 数据量：{data.get('volume')}
        - 路段：{data.get('route')}

        请识别：
        1. 时间规律（高峰期、周期性）
        2. 空间规律（瓶颈位置）
        3. 异常模式
        """)

        return {
            'temporal_patterns': '每日17:00-19:00出现高峰',
            'spatial_patterns': 'KP20-30为主要瓶颈',
            'anomalies': '节假日前后流量异常',
            'llm_analysis': thought
        }

    def extract_features(self, data: Dict) -> List[Dict]:
        """提取关键特征"""
        return [
            {'type': 'peak_hours', 'value': [17, 18, 19]},
            {'type': 'bottleneck_locations', 'value': [20, 25, 30]},
            {'type': 'average_duration', 'value': 120}
        ]

    def compute_statistics(self, data: Dict) -> Dict:
        """计算统计信息"""
        return {
            'total_events': 150,
            'average_severity': 0.7,
            'max_congestion_length': 15
        }

class PatternRecognitionAgent(BaseAgent):
    """模式识别Agent - 负责识别拥堵模式和重叠区域"""

    def __init__(self, llm=None):
        super().__init__("PatternExpert", "模式识别专家", llm)

    def process(self, message: Message) -> Message:
        """处理模式识别请求"""
        if message.type == MessageType.REQUEST:
            triangles = message.content.get('triangles', [])
            features = message.content.get('features', {})

            # 识别重叠模式
            overlaps = self.identify_overlaps(triangles)

            # 分类拥堵类型
            congestion_types = self.classify_congestion(triangles, features)

            # 找出关键模式
            key_patterns = self.extract_key_patterns(overlaps, congestion_types)

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content={
                    'overlaps': overlaps,
                    'congestion_types': congestion_types,
                    'key_patterns': key_patterns,
                    'confidence': self.calculate_confidence(overlaps)
                }
            )

    def identify_overlaps(self, triangles: List[Dict]) -> List[Dict]:
        """识别三角形重叠区域"""
        thought = self.think(f"""
        分析{len(triangles)}个历史拥堵三角形的重叠情况：

        需要考虑：
        1. 时间维度的重叠（相同时段）
        2. 空间维度的重叠（相同路段）
        3. 重叠的频率和规律
        """)

        # 模拟重叠分析
        overlaps = []
        for i in range(3):  # 假设找到3个主要重叠区域
            overlaps.append({
                'region_id': i,
                'time_range': f"{16+i}:00-{18+i}:00",
                'location_range': f"KP{20+i*5}-{25+i*5}",
                'frequency': 5 - i,
                'confidence': 0.9 - i * 0.1
            })
        return overlaps

    def classify_congestion(self, triangles: List, features: Dict) -> Dict:
        """分类拥堵类型"""
        return {
            'recurrent': '每日高峰期拥堵',
            'incident': '事故导致拥堵',
            'seasonal': '节假日拥堵'
        }

    def extract_key_patterns(self, overlaps: List, types: Dict) -> List[Dict]:
        """提取关键模式"""
        return [
            {
                'pattern': '工作日晚高峰',
                'location': 'KP20-30',
                'time': '17:00-19:00',
                'probability': 0.85
            }
        ]

    def calculate_confidence(self, overlaps: List) -> float:
        """计算置信度"""
        if overlaps:
            return sum(o.get('confidence', 0) for o in overlaps) / len(overlaps)
        return 0.5

class PredictionExpertAgent(BaseAgent):
    """预测专家Agent - 负责生成预测"""

    def __init__(self, llm=None):
        super().__init__("PredictionExpert", "交通预测专家", llm)

    def process(self, message: Message) -> Message:
        """生成预测"""
        if message.type == MessageType.REQUEST:
            patterns = message.content.get('patterns', {})
            overlaps = message.content.get('overlaps', [])

            # 基于模式生成预测
            prediction = self.generate_prediction(patterns, overlaps)

            # 生成预测三角形
            predicted_triangles = self.create_prediction_triangles(prediction)

            # 解释预测理由
            explanation = self.explain_prediction(prediction, patterns)

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content={
                    'prediction': prediction,
                    'triangles': predicted_triangles,
                    'explanation': explanation,
                    'confidence_level': self.assess_confidence(prediction)
                }
            )

    def generate_prediction(self, patterns: Dict, overlaps: List) -> Dict:
        """生成预测结果"""
        thought = self.think(f"""
        基于以下信息进行预测：
        - 识别到的模式：{patterns}
        - 重叠区域：{len(overlaps)}个

        考虑因素：
        1. 历史规律的延续性
        2. 今年的特殊因素
        3. 趋势变化

        生成预测...
        """)

        return {
            'predicted_date': '2024-05-15',
            'predicted_time_range': '17:00-19:30',
            'predicted_location': 'KP20-35',
            'severity': 'high',
            'duration_minutes': 150,
            'affected_distance_km': 15
        }

    def create_prediction_triangles(self, prediction: Dict) -> List[Dict]:
        """创建预测三角形"""
        return [{
            'apex': {'time': 110, 'location': 27.5},  # 顶点
            'base_start': {'time': 0, 'location': 20},
            'base_end': {'time': 0, 'location': 35},
            'height': 150,  # 持续时间
            'confidence': 0.8
        }]

    def explain_prediction(self, prediction: Dict, patterns: Dict) -> str:
        """解释预测理由"""
        return f"""
        预测理由：
        1. 基于过去3年数据，该路段在17:00-19:00期间有85%概率发生拥堵
        2. KP20-35是识别出的主要瓶颈区域
        3. 预测的拥堵模式与历史模式高度吻合
        4. 考虑了季节性因素和节假日影响
        """

    def assess_confidence(self, prediction: Dict) -> float:
        """评估预测置信度"""
        return 0.82

class ValidationAgent(BaseAgent):
    """验证Agent - 负责验证预测的合理性"""

    def __init__(self, llm=None):
        super().__init__("Validator", "预测验证专家", llm)

    def process(self, message: Message) -> Message:
        """验证预测结果"""
        if message.type == MessageType.REQUEST:
            prediction = message.content.get('prediction', {})
            historical = message.content.get('historical_data', {})

            # 验证合理性
            validation_result = self.validate_prediction(prediction, historical)

            # 提出改进建议
            suggestions = self.generate_suggestions(validation_result)

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.VALIDATION,
                content={
                    'is_valid': validation_result['is_valid'],
                    'issues': validation_result['issues'],
                    'suggestions': suggestions,
                    'confidence': validation_result['confidence']
                }
            )

    def validate_prediction(self, prediction: Dict, historical: Dict) -> Dict:
        """验证预测的合理性"""
        thought = self.think(f"""
        验证以下预测是否合理：
        - 预测时间：{prediction.get('predicted_time_range')}
        - 预测位置：{prediction.get('predicted_location')}
        - 严重程度：{prediction.get('severity')}

        检查：
        1. 是否符合历史规律
        2. 是否有物理约束冲突
        3. 是否考虑了所有因素
        """)

        issues = []

        # 模拟验证逻辑
        if prediction.get('duration_minutes', 0) > 300:
            issues.append("预测持续时间过长，历史最长为240分钟")

        if prediction.get('affected_distance_km', 0) > 20:
            issues.append("影响范围可能过大")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'confidence': 0.75 if len(issues) == 0 else 0.5,
            'llm_validation': thought
        }

    def generate_suggestions(self, validation_result: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if not validation_result['is_valid']:
            for issue in validation_result['issues']:
                if "持续时间" in issue:
                    suggestions.append("建议重新评估拥堵持续时间，参考历史平均值")
                if "影响范围" in issue:
                    suggestions.append("建议缩小预测影响范围至15km以内")

        suggestions.append("建议增加对特殊事件的考虑（如大型活动、施工等）")

        return suggestions

class OrchestratorAgent(BaseAgent):
    """协调Agent - 负责协调其他Agent的工作"""

    def __init__(self, agents: Dict[str, BaseAgent], llm=None):
        super().__init__("Orchestrator", "首席协调官", llm)
        self.agents = agents
        self.workflow_state = {}
        self.decision_history = []

    def orchestrate(self, user_input: str, data: Dict) -> Dict:
        """协调整个预测流程"""
        print(f"🎯 开始交通预测分析流程...")
        print(f"📝 用户需求：{user_input}\n")

        # 步骤1：数据分析
        print("📊 Step 1: 数据分析")
        data_analysis = self.request_analysis(data)
        print(f"  ✅ 发现 {len(data_analysis.get('features', []))} 个关键特征\n")

        # 步骤2：模式识别
        print("🔍 Step 2: 模式识别")
        patterns = self.request_pattern_recognition(data_analysis)
        print(f"  ✅ 识别到 {len(patterns.get('overlaps', []))} 个重叠区域\n")

        # 步骤3：生成预测
        print("🔮 Step 3: 生成预测")
        prediction = self.request_prediction(patterns, data_analysis)
        print(f"  ✅ 预测完成，置信度：{prediction.get('confidence_level', 0):.2%}\n")

        # 步骤4：验证预测
        print("✅ Step 4: 验证预测")
        validation = self.validate_prediction(prediction, data)

        # 步骤5：迭代优化（如果需要）
        final_prediction = prediction
        if not validation['is_valid']:
            print(f"  ⚠️ 发现 {len(validation['issues'])} 个问题，进行优化...")
            final_prediction = self.refine_prediction(prediction, validation)
            print("  ✅ 预测已优化\n")
        else:
            print("  ✅ 预测验证通过\n")

        # 生成最终报告
        report = self.generate_final_report(
            data_analysis, patterns, final_prediction, validation
        )

        return {
            'success': True,
            'prediction': final_prediction,
            'report': report,
            'confidence': final_prediction.get('confidence_level', 0),
            'workflow_log': self.decision_history
        }

    def request_analysis(self, data: Dict) -> Dict:
        """请求数据分析"""
        message = Message(
            sender=self.name,
            receiver="DataAnalyst",
            type=MessageType.REQUEST,
            content={'data': data}
        )
        response = self.agents['data_analyst'].process(message)
        self.decision_history.append(f"数据分析完成：{response.content.get('analysis')}")
        return response.content

    def request_pattern_recognition(self, analysis: Dict) -> Dict:
        """请求模式识别"""
        message = Message(
            sender=self.name,
            receiver="PatternExpert",
            type=MessageType.REQUEST,
            content={
                'triangles': analysis.get('triangles', []),
                'features': analysis.get('features', {})
            }
        )
        response = self.agents['pattern_expert'].process(message)
        self.decision_history.append(f"模式识别完成：发现{len(response.content.get('key_patterns', []))}个关键模式")
        return response.content

    def request_prediction(self, patterns: Dict, analysis: Dict) -> Dict:
        """请求生成预测"""
        message = Message(
            sender=self.name,
            receiver="PredictionExpert",
            type=MessageType.REQUEST,
            content={
                'patterns': patterns,
                'overlaps': patterns.get('overlaps', []),
                'features': analysis.get('features', {})
            }
        )
        response = self.agents['prediction_expert'].process(message)
        self.decision_history.append(f"预测生成：{response.content.get('prediction')}")
        return response.content

    def validate_prediction(self, prediction: Dict, historical_data: Dict) -> Dict:
        """验证预测"""
        message = Message(
            sender=self.name,
            receiver="Validator",
            type=MessageType.REQUEST,
            content={
                'prediction': prediction.get('prediction', {}),
                'historical_data': historical_data
            }
        )
        response = self.agents['validator'].process(message)
        return response.content

    def refine_prediction(self, prediction: Dict, validation: Dict) -> Dict:
        """根据验证结果优化预测"""
        refined = prediction.copy()

        # 根据建议调整
        for suggestion in validation.get('suggestions', []):
            if "持续时间" in suggestion:
                refined['prediction']['duration_minutes'] = min(
                    refined['prediction']['duration_minutes'], 240
                )
            if "影响范围" in suggestion:
                refined['prediction']['affected_distance_km'] = min(
                    refined['prediction']['affected_distance_km'], 15
                )

        refined['confidence_level'] = refined.get('confidence_level', 0.5) * 0.9
        self.decision_history.append("预测已根据验证结果优化")

        return refined

    def generate_final_report(self, analysis: Dict, patterns: Dict,
                            prediction: Dict, validation: Dict) -> str:
        """生成最终报告"""
        report = f"""
# 交通拥堵预测报告

## 1. 数据分析摘要
- 分析时间范围：{analysis.get('statistics', {}).get('time_range', '2021-2023')}
- 总拥堵事件数：{analysis.get('statistics', {}).get('total_events', 0)}
- 主要瓶颈位置：{', '.join([f['value'] for f in analysis.get('features', []) if f['type'] == 'bottleneck_locations'][:1])}

## 2. 识别的关键模式
{self.format_patterns(patterns)}

## 3. 预测结果
- **预测日期**：{prediction.get('prediction', {}).get('predicted_date')}
- **预测时段**：{prediction.get('prediction', {}).get('predicted_time_range')}
- **预测位置**：{prediction.get('prediction', {}).get('predicted_location')}
- **严重程度**：{prediction.get('prediction', {}).get('severity')}
- **持续时间**：{prediction.get('prediction', {}).get('duration_minutes')}分钟
- **影响范围**：{prediction.get('prediction', {}).get('affected_distance_km')}公里
- **置信度**：{prediction.get('confidence_level', 0):.2%}

## 4. 预测理由
{prediction.get('explanation', '基于历史数据分析')}

## 5. 验证结果
- 验证状态：{'✅ 通过' if validation.get('is_valid') else '⚠️ 需要注意'}
- 置信度：{validation.get('confidence', 0):.2%}
{self.format_issues(validation.get('issues', []))}

## 6. 建议
{self.format_suggestions(validation.get('suggestions', []))}

---
*报告生成时间：2024-11-15*
*Multi-Agent交通预测系统 v1.0*
        """
        return report

    def format_patterns(self, patterns: Dict) -> str:
        """格式化模式信息"""
        result = []
        for pattern in patterns.get('key_patterns', []):
            result.append(f"- {pattern['pattern']}：{pattern['location']}，{pattern['time']}")
        return '\n'.join(result) if result else "- 无特殊模式"

    def format_issues(self, issues: List) -> str:
        """格式化问题列表"""
        if not issues:
            return ""
        return "\n### 发现的问题：\n" + '\n'.join(f"- {issue}" for issue in issues)

    def format_suggestions(self, suggestions: List) -> str:
        """格式化建议列表"""
        return '\n'.join(f"- {s}" for s in suggestions)


def create_multi_agent_system(llm=None):
    """创建多Agent系统"""
    # 创建各个专业Agent
    agents = {
        'data_analyst': DataAnalystAgent(llm),
        'pattern_expert': PatternRecognitionAgent(llm),
        'prediction_expert': PredictionExpertAgent(llm),
        'validator': ValidationAgent(llm)
    }

    # 创建协调者
    orchestrator = OrchestratorAgent(agents, llm)

    return orchestrator


if __name__ == "__main__":
    # 示例：运行多Agent系统
    print("=" * 50)
    print("🚦 多Agent交通预测系统")
    print("=" * 50 + "\n")

    # 创建系统
    system = create_multi_agent_system()

    # 模拟输入数据
    user_input = "请预测関越高速公路2024年5月的拥堵情况"
    mock_data = {
        'route': '関越高速',
        'time_range': '2021-2023',
        'volume': 450,
        'triangles': [
            {'id': 1, 'time': 100, 'location': 25},
            {'id': 2, 'time': 105, 'location': 27},
            {'id': 3, 'time': 110, 'location': 23},
        ]
    }

    # 运行预测流程
    result = system.orchestrate(user_input, mock_data)

    # 输出结果
    print("=" * 50)
    print("📋 最终预测报告")
    print("=" * 50)
    print(result['report'])

    print("\n" + "=" * 50)
    print("🎯 预测成功完成！")
    print(f"总体置信度：{result['confidence']:.2%}")
    print("=" * 50)