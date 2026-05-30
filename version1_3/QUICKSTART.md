# 🚀 快速开始指南

## 📋 前提条件

1. Python 3.8+
2. 已安装必要的依赖包：
   ```bash
   pip install openai pandas numpy matplotlib langgraph
   ```

3. OpenAI API密钥（可选，没有会使用fallback模式）

## ⚡ 5分钟快速上手

### 步骤1: 配置API密钥（可选）

```bash
# 方法1: 设置环境变量（推荐）
export OPENAI_API_KEY='your-api-key-here'

# 方法2: 或在Python代码中设置
```

```python
from config import config
config.set_openai_api_key('your-api-key-here')
```

### 步骤2: 运行工作流

```python
from implementation import compiled_agent

# 分析交通数据
result = compiled_agent.invoke({
    "user_input": "请分析関越高速公路上行方向2024年的交通情况"
})

# 查看结果
print("="*70)
print("LLM分析结果:")
print("="*70)
print(result.get('llm_analysis'))

print("\n" + "="*70)
print("最终报告:")
print("="*70)
print(result.get('final_report'))
```

### 步骤3: 查看输出

工作流会自动：
1. ✅ 解析用户输入（chatbot节点）
2. ✅ 加载和可视化数据（visualization节点）
3. ✅ 使用LLM分析热点（analyze_with_llm节点）
4. ✅ 生成分析报告（report节点）

输出文件位置：
- 可视化图片: `output/triangles_*.png`
- CSV预测结果: `output/congestion_prediction_*.csv`
- 最终报告: 存储在 `result['final_report']` 中

## 🎨 自定义使用

### 场景1: 直接使用LLM分析器

```python
from llm_analyzer import create_batch_analyzer
from data_preparers import DataPreparer

# 准备数据
triangles = [
    {"kp_start": 10.0, "kp_end": 15.0, "peak_kp": 12.5,
     "time_start": 480, "time_end": 540, "peak_time": 510},
    # ... 更多三角形
]

# 创建分析器
analyzer = create_batch_analyzer(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# 执行分析
direction_data = {
    "上": {"triangles": triangles}
}

results = analyzer.analyze_all_directions(
    direction_data=direction_data,
    csv_files=["関越2024上り.csv"]
)

print(results)
```

### 场景2: 只使用Fallback分析器

```python
from fallback_analyzer import FallbackAnalyzer

# 创建fallback分析器
analyzer = FallbackAnalyzer(
    kp_threshold=5.0,
    time_threshold=180,
    min_cluster_size=2
)

# 分析
result = analyzer.analyze(
    triangles=triangles,
    direction="上"
)

print(result)
```

### 场景3: 自定义Prompt

```python
from prompt_templates import build_analysis_prompt

# 构建自定义prompt
prompt = build_analysis_prompt(
    direction="上",
    csv_files=["関越2024上り.csv"],
    triangles=triangles
)

# 查看生成的prompt
print(prompt)

# 然后可以手动调用LLM或修改prompt内容
```

## 🔧 常见问题

### Q1: 没有API密钥怎么办？

**A**: 系统会自动使用fallback模式（基于规则的分析），虽然效果不如LLM，但仍能识别基本的热点。

```python
# 强制使用fallback模式
from config import config
config.use_real_llm = False
```

### Q2: LLM分析太慢怎么办？

**A**:
1. 使用更快的模型（如gpt-4o-mini）
2. 增加timeout设置
3. 减少输入数据量

```python
from llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(
    api_key="your-key",
    model="gpt-4o-mini",  # 更快的模型
    timeout=60,           # 增加超时
    max_tokens=2000       # 减少token
)
```

### Q3: 如何调试LLM输入/输出？

**A**: 在 `llm_analyzer.py` 中添加调试代码：

```python
# 在 _call_llm_api 方法中
def _call_llm_api(self, ...):
    user_prompt = build_analysis_prompt(...)

    # 添加这些调试代码
    print("=== LLM输入Prompt ===")
    print(user_prompt[:500])  # 打印前500字符

    response = self.client.chat.completions.create(...)
    result_text = response.choices[0].message.content

    print("=== LLM输出 ===")
    print(result_text)

    return json.loads(result_text)
```

### Q4: 如何修改热点识别标准？

**A**: 编辑 `prompt_templates.py` 中的prompt：

```python
# 在 get_hotspot_analysis_prompt 方法中修改这部分
判断标准：
1. 空间聚集性: 多个事件在5-10km范围内  # ← 修改这里
2. 时间聚集性: 在2-4小时范围内         # ← 修改这里
3. 频次要求: 至少2-3个事件            # ← 修改这里
```

### Q5: 如何切换到不同的LLM模型？

**A**:
```python
# 方法1: 在config.py中全局修改
self.openai_model = "gpt-4o"  # 或其他模型

# 方法2: 创建分析器时指定
from llm_analyzer import create_batch_analyzer
analyzer = create_batch_analyzer(
    api_key="your-key",
    model="gpt-4o"  # 指定模型
)
```

## 📊 输出示例

### LLM分析结果示例

```json
{
  "上": {
    "direction": "上",
    "hotspots": [
      {
        "hotspot_id": 1,
        "kp_range": [23.5, 31.2],
        "time_range": [960, 1200],
        "included_triangle_ids": [1, 3, 5, 8],
        "frequency": 4,
        "severity": "high",
        "description": "晚高峰KP23.5-31.2区间拥堵"
      }
    ],
    "summary": {
      "total_hotspots": 1,
      "most_severe_hotspot_id": 1,
      "analysis_confidence": 0.85
    }
  }
}
```

### 控制台输出示例

```
======================================================================
🤖 Starting LLM-based hotspot analysis
======================================================================

============================================================
Analyzing 上 direction...
============================================================

🤖 Analyzing 45 triangles for 上 direction using LLM...
✅ LLM analysis completed for 上 direction
  Identified 3 hotspots

======================================================================
📊 LLM Analysis Summary
======================================================================
  上方向: 3 hotspots (confidence: 0.85)

  Total hotspots identified: 3
======================================================================

=== LLM-based Hotspot Summary ===
上り方向: 3 hotspots, 45 events
  Most severe: KP 23.5-31.2 km (4 times)
  Confidence: 0.85
```

## 🎯 下一步

1. 阅读 `LLM_ANALYZER_README.md` 了解详细的模块文档
2. 查看 `implementation.py` 了解节点实现细节
3. 修改 `prompt_templates.py` 自定义分析逻辑
4. 调整 `config.py` 配置系统参数

## 💡 最佳实践

1. **开发时使用fallback模式**，避免频繁调用API产生费用
2. **生产环境使用LLM模式**，获得更好的分析效果
3. **定期备份重要的分析结果**
4. **监控API使用量和成本**
5. **保持模块化**，方便后续扩展和维护

## 📚 相关文档

- `LLM_ANALYZER_README.md` - 详细模块文档
- `README.md` - 项目总体说明
- `COMMANDS.md` - 命令行使用说明

---

*祝使用愉快！如有问题欢迎反馈。*