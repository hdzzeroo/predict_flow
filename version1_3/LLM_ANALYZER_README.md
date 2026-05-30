# LLM分析器模块使用文档

## 📖 概述

本项目已将原有的 `cluster` 和 `draw_hulls` 两个节点合并为一个新的 `analyze_with_llm` 节点，使用大语言模型（LLM）来智能分析交通拥堵热点。

## 🏗️ 模块结构

```
version1_3/
├── llm_analyzer.py          # LLM分析核心逻辑
├── data_preparers.py        # 数据准备和预处理
├── prompt_templates.py      # Prompt模板管理
├── fallback_analyzer.py     # Fallback分析器（无LLM时使用）
├── implementation.py        # 工作流节点实现
└── stub.py                  # 工作流定义
```

## 🔄 新的工作流

```
START → chatbot → visualization → analyze_with_llm → report → END
```

**对比旧流程**:
```
旧: START → chatbot → visualization → cluster → draw_hulls → report → END
新: START → chatbot → visualization → analyze_with_llm → report → END
```

## 📦 模块详解

### 1. `prompt_templates.py` - Prompt模板模块

负责管理所有LLM相关的prompt模板。

**主要类**:
- `PromptTemplates`: 提供系统prompt和用户prompt的生成

**主要函数**:
```python
# 构建分析prompt
prompt = build_analysis_prompt(
    direction="上",
    csv_files=["関越2024上り.csv"],
    triangles=triangle_list
)
```

**如何修改Prompt**:
1. 打开 `prompt_templates.py`
2. 修改 `get_hotspot_analysis_prompt()` 方法
3. 调整分析要求、输出格式等

### 2. `data_preparers.py` - 数据准备模块

负责将原始数据转换为LLM可理解的格式。

**主要类**:
- `DataPreparer`: 准备和标准化三角形数据
- `RawDataLoader`: 加载CSV文件摘要
- `OutputFormatter`: 格式化输出结果

**主要函数**:
```python
# 准备LLM输入数据
llm_input = DataPreparer.prepare_llm_input(
    triangles=triangles,
    direction="上",
    csv_files=["file1.csv", "file2.csv"]
)

# 格式化热点信息用于显示
formatted = OutputFormatter.format_hotspot_for_display(hotspot)
```

**如何扩展**:
- 在 `DataPreparer` 中添加新的数据处理方法
- 在 `OutputFormatter` 中添加新的格式化方法

### 3. `fallback_analyzer.py` - Fallback分析器

当LLM不可用时，使用简单的规则算法进行分析。

**主要类**:
- `FallbackAnalyzer`: 基于距离的简单聚类分析器

**配置参数**:
```python
analyzer = FallbackAnalyzer(
    kp_threshold=5.0,      # KP距离阈值（km）
    time_threshold=180,     # 时间距离阈值（分钟）
    min_cluster_size=2      # 最小聚类大小
)
```

**如何调整**:
1. 打开 `fallback_analyzer.py`
2. 修改 `__init__` 中的默认参数
3. 或在创建时传入自定义参数

### 4. `llm_analyzer.py` - LLM分析核心

负责调用OpenAI API进行分析。

**主要类**:
- `LLMAnalyzer`: 单方向分析器
- `BatchLLMAnalyzer`: 批量分析器（支持多方向）

**使用示例**:
```python
from llm_analyzer import create_batch_analyzer

# 创建批量分析器
analyzer = create_batch_analyzer(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# 分析所有方向
results = analyzer.analyze_all_directions(
    direction_data=direction_data,
    csv_files=csv_files
)
```

**配置选项**:
```python
analyzer = LLMAnalyzer(
    api_key="your-api-key",
    model="gpt-4o-mini",        # 模型选择
    temperature=0.1,             # 温度参数
    max_tokens=4000,             # 最大token数
    timeout=30,                  # 超时时间
    use_fallback=True            # 是否使用fallback
)
```

## 🔧 常见修改场景

### 场景1: 修改分析逻辑（调整Prompt）

```python
# 在 prompt_templates.py 中修改
def get_hotspot_analysis_prompt(...):
    prompt = f"""
    你是交通分析专家...

    ## 分析任务
    请识别拥堵热点，标准是：
    1. 空间聚集性: 5-10km范围  # ← 修改这里
    2. 时间聚集性: 2-4小时范围  # ← 修改这里
    3. 最少事件数: 2-3个        # ← 修改这里
    ...
    """
```

### 场景2: 更换LLM模型

```python
# 在 config.py 中修改
self.openai_model = "gpt-4o"  # 改为gpt-4o或其他模型

# 或在使用时指定
analyzer = create_batch_analyzer(
    api_key=api_key,
    model="gpt-4o"  # 使用更强大的模型
)
```

### 场景3: 调整Fallback算法参数

```python
# 在 fallback_analyzer.py 中修改默认值
def __init__(
    self,
    kp_threshold: float = 10.0,    # 改为10km
    time_threshold: int = 240,      # 改为4小时
    min_cluster_size: int = 3       # 改为至少3个事件
):
    ...
```

### 场景4: 添加新的数据统计

```python
# 在 data_preparers.py 的 DataPreparer 类中添加
@staticmethod
def calculate_density_statistics(triangles: List[Dict]) -> Dict:
    """计算拥堵密度统计"""
    # 你的自定义统计逻辑
    return {
        "density": ...,
        "peak_density": ...
    }
```

### 场景5: 修改输出格式

```python
# 在 data_preparers.py 的 OutputFormatter 类中修改
@staticmethod
def format_hotspot_for_display(hotspot: Dict) -> str:
    # 自定义你的输出格式
    return f"热点{hotspot['hotspot_id']}: ..."
```

## 📊 LLM输出格式

LLM严格按照以下JSON格式输出：

```json
{
    "direction": "上",
    "hotspots": [
        {
            "hotspot_id": 1,
            "kp_range": [23.5, 31.2],
            "time_range": [960, 1200],
            "included_triangle_ids": [1, 3, 5, 8],
            "frequency": 4,
            "severity": "high",
            "description": "晚高峰拥堵热点"
        }
    ],
    "summary": {
        "total_hotspots": 1,
        "most_severe_hotspot_id": 1,
        "analysis_confidence": 0.85
    }
}
```

## ⚙️ 配置说明

### OpenAI API配置

在 `config.py` 中配置：

```python
# 方法1: 设置环境变量
export OPENAI_API_KEY='your-api-key-here'

# 方法2: 在代码中设置
from config import config
config.set_openai_api_key('your-api-key-here')

# 方法3: 交互式设置
from config import setup_api_key
setup_api_key()
```

### 模型选择

支持的模型：
- `gpt-4o-mini`: 推荐，性价比高
- `gpt-4o`: 更强大，但成本较高
- `gpt-3.5-turbo`: 便宜，但效果可能较差

## 🧪 测试

### 基本测试

```python
# 测试单个模块
python -c "from llm_analyzer import create_batch_analyzer; print('OK')"
python -c "from data_preparers import DataPreparer; print('OK')"
python -c "from prompt_templates import PromptTemplates; print('OK')"
python -c "from fallback_analyzer import FallbackAnalyzer; print('OK')"
```

### 完整工作流测试

```python
from implementation import compiled_agent

result = compiled_agent.invoke({
    "user_input": "请分析関越高速公路上行方向2024年的交通情况"
})

print(result.get('llm_analysis'))
```

## 🔍 调试技巧

### 1. 查看LLM输入

在 `llm_analyzer.py` 的 `_call_llm_api` 方法中添加：

```python
def _call_llm_api(...):
    user_prompt = build_analysis_prompt(...)
    print("=== LLM输入 ===")
    print(user_prompt)
    print("="*50)
    ...
```

### 2. 查看LLM输出

```python
result_text = response.choices[0].message.content
print("=== LLM输出 ===")
print(result_text)
print("="*50)
result = json.loads(result_text)
```

### 3. 强制使用Fallback模式

```python
# 临时禁用LLM，测试fallback
config.use_real_llm = False
```

## ⚠️ 注意事项

1. **API密钥安全**: 不要将API密钥提交到代码仓库
2. **成本控制**: gpt-4o成本较高，建议开发时使用gpt-4o-mini
3. **超时设置**: 处理大量数据时适当增加timeout
4. **Fallback可用性**: 确保fallback逻辑始终可用，作为备份方案
5. **JSON格式验证**: LLM输出会自动验证，无效输出会触发fallback

## 📝 更新日志

### v1.3 (当前版本)
- ✅ 用 `analyze_with_llm` 替换 `cluster` 和 `draw_hulls`
- ✅ 模块化设计，易于维护和扩展
- ✅ 支持LLM和fallback双模式
- ✅ 完善的错误处理和日志输出

### 与v1.2的区别
- 移除了DBSCAN聚类算法
- 移除了凸包大三角形计算
- 使用LLM智能识别热点
- 报告中显示LLM分析置信度

## 🤝 贡献

如果你想添加新功能或改进现有模块：

1. 在对应模块中添加新的类或函数
2. 保持模块化设计原则
3. 添加适当的文档字符串
4. 更新本README文档

## 📧 支持

如有问题，请查看：
1. 本README文档
2. 各模块的文档字符串
3. `implementation.py` 中的节点实现