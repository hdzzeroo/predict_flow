# 多Agent交通预测系统 - AI开发指导文档

## 📋 项目概述

本文档指导AI助手完成一个基于LLM的多Agent交通预测系统，该系统模拟人类专家通过三角形可视化进行拥堵预测的工作流程。

### 核心目标
- 构建4个Agent的协作系统：协调器、数据提取、预测专家、验证
- 利用LLM视觉理解和推理能力进行预测
- 基于历史三角形重叠模式预测未来拥堵

## ⚠️ 严格遵守的开发原则

### 1. 绝对禁止的失败处理
```python
# ❌ 绝对禁止 - 永远不要隐藏错误
try:
    result = llm_call(prompt)
except:
    result = "默认值"  # 这会隐藏真正的问题！

# ✅ 正确做法 - 明确处理错误
try:
    result = llm_call(prompt)
    if not result or result.strip() == "":
        raise ValueError("LLM返回空结果，无法继续处理")
    if not validate_llm_output(result):
        raise ValueError(f"LLM输出格式无效: {result}")
except Exception as e:
    logger.error(f"LLM调用失败: {e}")
    raise  # 向上传播，让调用者决定如何处理
```

### 2. 强制模块化设计
每个组件必须：
- 可独立实例化和测试
- 支持配置启用/禁用
- 使用策略模式支持算法替换

```python
# 强制要求的接口设计
class AgentInterface(ABC):
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """每个Agent必须实现此方法"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict) -> bool:
        """输入验证必须实现"""
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """状态检查必须实现"""
        pass
```

### 3. 数据验证原则
所有数据传递都必须验证：
```python
def validate_triangle_data(triangle: Dict) -> bool:
    required_fields = ['vertices', 'center', 'area', 'kp_start', 'kp_end']
    if not all(field in triangle for field in required_fields):
        raise ValueError(f"三角形数据缺少必要字段: {required_fields}")

    # 物理约束验证
    if triangle['kp_end'] <= triangle['kp_start']:
        raise ValueError("KP结束位置必须大于起始位置")

    if triangle['area'] <= 0:
        raise ValueError("三角形面积必须为正数")

    return True
```

## 🏗️ 系统架构要求

### 1. 项目结构
```
src/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # 抽象基类
│   ├── orchestrator.py         # 协调器Agent
│   ├── data_extractor.py       # 数据提取Agent
│   ├── prediction_expert.py    # 预测专家Agent
│   └── validator.py            # 验证Agent
├── core/
│   ├── __init__.py
│   ├── data_structures.py      # 数据结构定义
│   ├── config.py              # 配置管理
│   └── exceptions.py          # 自定义异常
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # 三角形可视化
│   ├── data_loader.py         # 数据加载工具
│   └── validation.py          # 验证工具
├── llm/
│   ├── __init__.py
│   ├── client.py              # LLM客户端
│   └── prompts.py             # Prompt模板
└── tests/
    ├── test_agents.py
    ├── test_integration.py
    └── test_data/
```

### 2. 核心数据结构（基于DATA_SPECIFICATION.md）
```python
@dataclass
class TrafficEvent:
    """原始交通事件数据"""
    date: datetime.date
    direction: str  # "上" or "下"
    cause: str     # "交通集中"等
    road_name: str # "東北道"
    start_time: datetime.time    # 発生時刻
    peak_time: datetime.time     # ピーク時刻
    peak_length: float          # ピーク長(km)
    start_kp: float            # 発生Ｋｐ
    initial_length: float      # 発生時渋滞長
    duration: int              # 渋滞時間(分钟)

    def validate(self) -> None:
        """数据验证 - 必须实现"""
        if self.peak_time < self.start_time:
            raise ValueError("峰值时间不能早于开始时间")
        if self.peak_length < self.initial_length:
            raise ValueError("峰值拥堵长度不能小于初始长度")
        if self.direction not in ["上", "下"]:
            raise ValueError(f"方向必须是'上'或'下'，当前值: {self.direction}")

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
            raise ValueError("三角形必须有3个顶点")
        if self.area <= 0:
            raise ValueError("三角形面积必须为正数")
        if self.kp_end <= self.kp_start:
            raise ValueError("结束KP必须大于起始KP")

@dataclass
class PredictionResult:
    """预测结果"""
    prediction_id: str
    target_date: datetime.date
    direction: str
    predicted_triangles: List[Triangle]
    time_range: Tuple[datetime.time, datetime.time]  # (start, end)
    location_range: Tuple[float, float]              # (start_kp, end_kp)
    peak_time: datetime.time
    peak_location: float
    severity: str        # "high", "medium", "low"
    confidence: float    # 0-1
    explanation: str     # LLM生成的预测理由
    historical_basis: List[Triangle]  # 预测基于的历史三角形
```

## 🤖 Agent实现要求

### 1. 协调器Agent (OrchestratorAgent)
```python
class OrchestratorAgent(AgentInterface):
    """总协调Agent - 负责流程控制"""

    def __init__(self, config: Dict):
        self.config = config
        self.agents = self._initialize_agents()
        self.logger = setup_logger("orchestrator")

    def process(self, request: Dict) -> Dict:
        """
        处理预测请求的主流程

        Args:
            request: {
                'target_date': '2024-05-15',
                'road_name': '東北道',
                'direction': '上',  # 可选
                'time_range': '17:00-19:00'  # 可选
            }

        Returns:
            完整的预测结果和执行日志
        """
        try:
            # 1. 验证输入
            self.validate_input(request)

            # 2. 数据提取
            historical_data = self.agents['data_extractor'].process({
                'target_date': request['target_date'],
                'road_name': request['road_name'],
                'direction': request.get('direction')
            })

            # 3. 预测生成
            prediction = self.agents['prediction_expert'].process(historical_data)

            # 4. 验证预测
            validated_result = self.agents['validator'].process({
                'prediction': prediction,
                'historical_data': historical_data
            })

            return validated_result

        except Exception as e:
            self.logger.error(f"协调器处理失败: {e}")
            raise

    def validate_input(self, request: Dict) -> bool:
        """严格的输入验证"""
        required_fields = ['target_date', 'road_name']
        for field in required_fields:
            if field not in request:
                raise ValueError(f"缺少必要字段: {field}")

        # 日期格式验证
        try:
            datetime.strptime(request['target_date'], '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式必须是YYYY-MM-DD")

        return True
```

### 2. 数据提取Agent (DataExtractionAgent)
```python
class DataExtractionAgent(AgentInterface):
    """数据提取Agent - 负责历史数据提取和预处理"""

    def process(self, input_data: Dict) -> Dict:
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
                raise ValueError(f"未找到相关历史数据: {input_data}")

            # 2. 转换为三角形数据
            triangles = self._convert_to_triangles(raw_data)

            if not triangles:
                raise ValueError("无法从历史数据生成有效的三角形")

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

    def _load_historical_data(self, target_date: str, road_name: str, direction: str = None) -> pd.DataFrame:
        """加载历史数据 - 基于DATA_SPECIFICATION.md的文件命名规则"""
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        historical_files = []

        # 查找过去3年同期数据
        for year_offset in [1, 2, 3]:
            historical_date = target_date_obj.replace(year=target_date_obj.year - year_offset)

            # 根据DATA_SPECIFICATION.md的命名规则
            if direction:
                pattern = f"{road_name}_{direction}_{historical_date.strftime('%Y_%m-%d')}.csv"
            else:
                # 如果未指定方向，查找两个方向的数据
                for dir in ['上', '下']:
                    pattern = f"{road_name}_{dir}_{historical_date.strftime('%Y_%m-%d')}.csv"
                    file_path = os.path.join(self.config['data_path'], pattern)
                    if os.path.exists(file_path):
                        historical_files.append(file_path)
                continue

            file_path = os.path.join(self.config['data_path'], pattern)
            if os.path.exists(file_path):
                historical_files.append(file_path)

        if not historical_files:
            raise FileNotFoundError(f"未找到{road_name}在{target_date}的历史数据文件")

        # 合并所有历史数据
        all_data = []
        for file_path in historical_files:
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                self.logger.warning(f"读取文件失败 {file_path}: {e}")

        if not all_data:
            raise ValueError("所有历史数据文件都无法读取")

        return pd.concat(all_data, ignore_index=True)
```

### 3. 预测专家Agent (PredictionExpertAgent)
```python
class PredictionExpertAgent(AgentInterface):
    """预测专家Agent - 核心LLM预测逻辑"""

    def __init__(self, llm_client, config: Dict):
        self.llm_client = llm_client
        self.config = config
        self.prompt_template = self._load_prediction_prompt()

    def process(self, input_data: Dict) -> Dict:
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

            return parsed_prediction

        except Exception as e:
            self.logger.error(f"预测生成失败: {e}")
            raise

    def _call_llm_with_vision(self, prompt: str, image: Image, context: Dict) -> str:
        """调用支持视觉的LLM"""
        try:
            response = self.llm_client.call_with_vision(
                prompt=prompt,
                image=image,
                max_tokens=2000,
                temperature=0.1  # 低温度保证一致性
            )

            if not response or response.strip() == "":
                raise ValueError("LLM返回空响应")

            # 检查响应是否包含必要信息
            if "预测三角形" not in response and "prediction" not in response.lower():
                raise ValueError(f"LLM响应似乎不包含预测信息: {response[:200]}")

            return response

        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise

    def _prepare_llm_input(self, raw_data: pd.DataFrame, triangles: List[Triangle], statistics: Dict) -> Dict:
        """准备LLM输入数据"""

        # 1. 格式化原始数据摘要
        data_summary = self._format_data_summary(raw_data, statistics)

        # 2. 格式化三角形坐标信息
        triangle_coords = self._format_triangle_coordinates(triangles)

        # 3. 构建提示词
        prompt = f"""
你是一位资深的交通拥堵预测专家。请基于以下三种数据进行预测：

## 1. 历史拥堵数据摘要
{data_summary}

## 2. 三角形坐标信息
{triangle_coords}

## 3. 可视化图像
[图像已加载，显示了历史三角形的分布和重叠情况]

## 预测任务
请观察图像中三角形的重叠模式，分析过去3年同期的拥堵规律，预测今年的拥堵情况。

## 输出要求（必须按此格式）
```json
{{
    "predicted_triangle": {{
        "apex_time": "18:15",
        "apex_kp": 27.5,
        "base_start_time": "17:00",
        "base_end_time": "19:30",
        "base_start_kp": 20.0,
        "base_end_kp": 35.0,
        "severity": "high",
        "confidence": 0.82
    }},
    "explanation": "基于过去3年数据，发现在该时段存在显著的重叠区域...",
    "historical_basis": ["2021年数据显示...", "2022年模式表明...", "2023年确认了..."]
}}
```

请确保预测结果具有物理合理性和逻辑一致性。
"""

        return {
            'prompt': prompt,
            'context': {
                'triangle_count': len(triangles),
                'date_range': statistics.get('date_range'),
                'road_name': statistics.get('road_name')
            }
        }
```

### 4. 验证Agent (ValidationAgent)
```python
class ValidationAgent(AgentInterface):
    """验证Agent - 负责预测结果的验证和质量控制"""

    def process(self, input_data: Dict) -> Dict:
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

            # 6. 如果验证失败，抛出详细错误
            if not all([physics_validation['passed'],
                       consistency_validation['passed'],
                       logic_validation['passed']]):
                failed_checks = []
                if not physics_validation['passed']:
                    failed_checks.extend(physics_validation['errors'])
                if not consistency_validation['passed']:
                    failed_checks.extend(consistency_validation['errors'])
                if not logic_validation['passed']:
                    failed_checks.extend(logic_validation['errors'])

                raise ValueError(f"预测验证失败: {'; '.join(failed_checks)}")

            return {
                'validated_prediction': prediction,
                'final_confidence': final_confidence,
                'validation_report': validation_report,
                'validation_details': {
                    'physics': physics_validation,
                    'consistency': consistency_validation,
                    'logic': logic_validation
                }
            }

        except Exception as e:
            self.logger.error(f"验证过程失败: {e}")
            raise

    def _validate_physics_constraints(self, prediction: Dict) -> Dict:
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

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
```

## 🔧 技术实现细节

### 1. LLM客户端实现
```python
class LLMClient:
    """LLM客户端 - 支持视觉输入"""

    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model = model
        self.client = self._initialize_client()

    def call_with_vision(self, prompt: str, image: Image, **kwargs) -> str:
        """带视觉输入的LLM调用"""
        try:
            # 将图像转换为base64
            image_base64 = self._image_to_base64(image)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.1)
            )

            result = response.choices[0].message.content

            if not result:
                raise ValueError("LLM返回空内容")

            return result

        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise
```

### 2. 可视化生成
```python
def generate_triangle_visualization(triangles: List[Triangle], config: Dict) -> Image:
    """生成三角形可视化图像"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置坐标轴
    ax.set_xlabel('位置 (KP)', fontsize=12)
    ax.set_ylabel('时间 (分钟)', fontsize=12)
    ax.set_title('历史拥堵三角形分布', fontsize=14)

    colors = plt.cm.Set3(np.linspace(0, 1, len(triangles)))

    for i, triangle in enumerate(triangles):
        # 绘制三角形
        vertices = np.array(triangle.vertices + [triangle.vertices[0]])  # 闭合
        ax.plot(vertices[:, 0], vertices[:, 1],
                color=colors[i], alpha=0.7, linewidth=2)
        ax.fill(vertices[:, 0], vertices[:, 1],
                color=colors[i], alpha=0.3)

        # 添加标注
        ax.annotate(f'T{i+1}', triangle.center,
                   ha='center', va='center', fontsize=8)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 保存为图像
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image
```

## 🧪 测试要求

### 1. 单元测试
每个Agent都必须有完整的单元测试：
```python
class TestDataExtractionAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DataExtractionAgent(test_config)
        self.sample_data = create_sample_traffic_data()

    def test_load_historical_data_success(self):
        """测试正常数据加载"""
        result = self.agent._load_historical_data(
            target_date='2024-05-15',
            road_name='東北道',
            direction='上'
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_load_historical_data_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            self.agent._load_historical_data(
                target_date='2099-99-99',  # 不存在的日期
                road_name='不存在的道路',
                direction='上'
            )

    def test_convert_to_triangles_validation(self):
        """测试三角形转换和验证"""
        triangles = self.agent._convert_to_triangles(self.sample_data)
        for triangle in triangles:
            triangle.validate()  # 必须通过验证
```

### 2. 集成测试
```python
class TestIntegration(unittest.TestCase):

    def test_full_prediction_workflow(self):
        """测试完整预测流程"""
        system = TrafficPredictionSystem(test_config)

        request = {
            'target_date': '2024-05-15',
            'road_name': '東北道',
            'direction': '上'
        }

        # 必须成功完成或明确失败
        try:
            result = system.predict(request)
            self.assertIn('validated_prediction', result)
            self.assertIn('final_confidence', result)
            self.assertGreater(result['final_confidence'], 0)
        except Exception as e:
            self.fail(f"预测流程应该成功或提供明确错误信息: {e}")
```

## 📊 监控和日志

### 1. 必须的日志记录
```python
import logging
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# 每个操作都必须记录
logger.info("开始数据提取", target_date=target_date, road_name=road_name)
logger.error("LLM调用失败", error=str(e), prompt_length=len(prompt))
logger.info("预测完成", confidence=result.confidence, duration=duration)
```

### 2. 性能监控
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__}执行成功", duration=duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__}执行失败", duration=duration, error=str(e))
            raise
    return wrapper
```

## ✅ 验收标准

### 1. 功能要求
- [ ] 能够加载和解析历史CSV数据
- [ ] 能够生成清晰的三角形可视化
- [ ] LLM能够基于视觉输入进行合理预测
- [ ] 预测结果通过所有验证检查
- [ ] 系统能够处理各种异常情况

### 2. 质量要求
- [ ] 测试覆盖率 > 80%
- [ ] 所有错误都有明确的错误信息
- [ ] 没有隐藏异常的代码
- [ ] 每个模块都可以独立运行
- [ ] 日志记录完整且结构化

### 3. 性能要求
- [ ] 单次预测完成时间 < 30秒
- [ ] 内存使用 < 2GB
- [ ] 能够处理3年历史数据
- [ ] LLM调用失败时有合理的重试机制

## 🚀 开发步骤建议

1. **第一步**：实现数据结构和验证逻辑
2. **第二步**：实现DataExtractionAgent（不依赖LLM）
3. **第三步**：实现可视化生成功能
4. **第四步**：实现LLM客户端和PredictionExpertAgent
5. **第五步**：实现ValidationAgent
6. **第六步**：实现OrchestratorAgent整合所有功能
7. **第七步**：编写完整测试套件
8. **第八步**：性能优化和错误处理完善

记住：**永远不要隐藏错误，宁可系统明确失败也不要给出错误的结果**！