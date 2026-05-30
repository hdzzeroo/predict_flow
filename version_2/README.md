# 多Agent交通拥堵预测系统 V2.0

基于LLM的多Agent系统，通过三角形可视化和模式识别来预测交通拥堵。

## 📋 项目概述

本系统模拟人类专家通过三角形可视化进行交通拥堵预测的工作流程，利用LLM的视觉理解和推理能力，无需复杂算法即可完成预测。

### 核心特性

- **4个Agent协作**：协调器、数据提取、预测专家、验证
- **LLM视觉理解**：支持图像输入的拥堵模式识别
- **三角形抽象**：将拥堵数据可视化为时空三角形
- **严格验证**：物理约束、历史一致性、逻辑合理性验证
- **完整监控**：结构化日志、性能监控、错误追踪

## 🏗️ 系统架构

```
用户请求 → 协调器Agent → 数据提取Agent → 预测专家Agent → 验证Agent → 预测结果
                ↓              ↓              ↓           ↓
            流程控制      历史数据处理    LLM视觉预测    结果验证
```

### Agent职责

- **协调器Agent (Orchestrator)**: 总控制，管理工作流程
- **数据提取Agent (DataExtraction)**: 历史数据提取和三角形生成
- **预测专家Agent (PredictionExpert)**: 基于LLM的视觉预测
- **验证Agent (Validation)**: 预测结果验证和质量控制

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
# 设置OpenAI API密钥（可选，测试时使用模拟LLM）
export OPENAI_API_KEY="your-api-key"

# 或者在config.yaml中配置
```

### 3. 运行示例

```bash
# 运行完整示例
python example_usage.py

# 运行测试
python run_tests.py
```

### 4. 基础使用

```python
from src.traffic_prediction_system import create_system

# 创建系统实例
system = create_system(use_mock_llm=True)  # 测试模式

# 执行预测
result = system.quick_predict('2024-05-15', '東北道', '上')

print(f"置信度: {result['confidence']:.2f}")
print(f"建议: {result['recommendation']}")
```

## 📁 项目结构

```
version_2/
├── src/
│   ├── agents/              # Agent实现
│   │   ├── base_agent.py    # 基础Agent类
│   │   ├── orchestrator.py  # 协调器Agent
│   │   ├── data_extractor.py # 数据提取Agent
│   │   ├── prediction_expert.py # 预测专家Agent
│   │   └── validator.py     # 验证Agent
│   ├── core/               # 核心数据结构
│   │   ├── data_structures.py # 数据结构定义
│   │   ├── config.py       # 配置管理
│   │   └── exceptions.py   # 自定义异常
│   ├── utils/              # 工具类
│   │   ├── visualization.py # 三角形可视化
│   │   ├── data_loader.py  # 数据加载
│   │   ├── validation.py   # 验证工具
│   │   └── logging_config.py # 日志配置
│   ├── llm/                # LLM集成
│   │   ├── client.py       # LLM客户端
│   │   └── prompts.py      # Prompt模板
│   ├── tests/              # 测试文件
│   └── traffic_prediction_system.py # 主系统入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── example_usage.py        # 使用示例
└── run_tests.py           # 测试运行器
```

## 📊 数据格式

### 输入数据结构（CSV）

```csv
date,上下,原因,道路番号,発生時刻,ピーク時刻,ピーク長,発生Ｋｐ,発生時渋滞長,渋滞時間
2023-05-15,上,交通集中,東北道,17:30:00,18:15:00,8.5,25.4,3.2,180
```

### 文件命名规则

```
{道路名}_{方向}_{年份}_{月份}-{日期}.csv
例：東北道_上_2023_05-15.csv
```

### 预测请求格式

```python
request = {
    'target_date': '2024-05-15',  # 预测日期
    'road_name': '東北道',        # 道路名称
    'direction': '上',           # 方向（可选）
    'time_range': '17:00-19:00'  # 时间范围（可选）
}
```

## 🔧 配置说明

### config.yaml

```yaml
# 数据路径
data_path: "./data"
output_path: "./output"

# 三角形生成配置
triangle_config:
  min_duration: 10        # 最小持续时间(分钟)
  min_length: 0.5         # 最小拥堵长度(km)
  time_resolution: 5      # 时间分辨率(分钟)
  merge_threshold: 0.8    # 合并阈值

# LLM配置
llm_config:
  model: "gpt-4-vision-preview"
  max_tokens: 2000
  temperature: 0.1
  timeout: 30

# 日志配置
enable_logging: true
log_level: "INFO"
```

## 🧪 测试

### 运行所有测试

```bash
python run_tests.py
```

### 运行特定测试

```bash
python run_tests.py test_agents
python run_tests.py test_integration
python run_tests.py test_data_structures
```

### 测试覆盖

- 单元测试：所有Agent和核心组件
- 集成测试：完整工作流程
- 性能测试：内存和时间限制
- 错误处理测试：各种异常情况

## 📈 监控和日志

### 日志类型

- **系统日志**: 组件状态和工作流程
- **性能日志**: 执行时间和内存使用
- **Agent日志**: 各Agent的处理过程
- **LLM日志**: API调用和响应
- **验证日志**: 验证过程和结果

### 性能监控

```python
from src.utils.logging_config import PerformanceLogger, LoggingContext

perf_logger = PerformanceLogger()
perf_logger.log_execution_time("operation_name", duration)

# 或使用上下文管理器
with LoggingContext(logger, "prediction_workflow") as ctx:
    result = system.predict(request)
    ctx.log_progress("预测完成", confidence=result['confidence'])
```

## 🚨 错误处理

系统遵循严格的错误处理原则：

- **绝不隐藏错误**：所有错误都明确报告
- **向上传播**：让调用者决定如何处理
- **详细日志**：记录错误上下文和调用栈
- **优雅降级**：在可能的情况下提供替代方案

### 常见错误类型

- `DataValidationError`: 数据验证失败
- `LLMError`: LLM API调用失败
- `AgentError`: Agent处理失败
- `ConfigurationError`: 配置无效

## 🔍 API参考

### 主要接口

```python
# 创建系统
system = TrafficPredictionSystem(config=config, use_mock_llm=False)

# 完整预测
result = system.predict({
    'target_date': '2024-05-15',
    'road_name': '東北道',
    'direction': '上'
})

# 快速预测
summary = system.quick_predict('2024-05-15', '東北道', '上')

# 系统状态
status = system.get_system_status()

# 配置验证
validation = system.validate_configuration()
```

### 预测结果结构

```python
{
    'request': {...},           # 原始请求
    'prediction': {             # 预测结果
        'predicted_triangle': {...},
        'explanation': '...',
        'confidence': 0.82
    },
    'validation': {             # 验证结果
        'validation_passed': True,
        'final_confidence': 0.78,
        'validation_report': '...'
    },
    'metadata': {...},          # 元数据
    'summary': {...}            # 摘要信息
}
```

## 🛠️ 开发指南

### 添加新Agent

1. 继承`BaseAgent`类
2. 实现必要的抽象方法
3. 添加单元测试
4. 在协调器中集成

### 自定义验证规则

1. 扩展`ValidationAgent`
2. 添加新的验证方法
3. 更新验证报告生成

### 扩展LLM支持

1. 实现新的LLM客户端
2. 遵循`LLMClient`接口
3. 添加相应的错误处理

## 📋 TODO清单

- [ ] 支持更多LLM提供商
- [ ] 实时数据流处理
- [ ] Web界面和API
- [ ] 模型性能评估
- [ ] 分布式部署支持

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 遵循代码规范
4. 添加测试
5. 提交Pull Request

## 📄 许可证

MIT License

## 📞 支持

如有问题请提交Issue或联系开发团队。

---

**版本**: 2.0.0
**更新日期**: 2024-11-15
**维护者**: Multi-Agent Traffic Prediction System Team