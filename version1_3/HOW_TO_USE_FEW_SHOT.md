# Few-Shot 示例使用指南

## 概述

本文档说明如何从 `workflow_state.json` 中提取LLM聚类结果，并将其作为few-shot示例添加到prompt中，以提高LLM的聚类质量。

## 已生成的文件

运行提取脚本后，会生成以下文件：

### 1. 原始数据提取
- **`output/few_shot_example.json`** (67KB)
  - 完整的结构化数据
  - 包含输入三角形和输出热点的所有细节
  - 适合程序化处理

- **`output/few_shot_example_readable.txt`** (8.2KB)
  - 人类可读的文本格式
  - 包含完整的输入、thinking、输出
  - 适合查看和理解示例

### 2. Few-Shot Prompt格式

#### 完整版 (~2200 tokens)
- **`output/few_shot_prompt_上.txt`** (上方向)
- **`output/few_shot_prompt_下.txt`** (下方向)

包含内容：
- ✅ 输入三角形数据（前10个）
- ✅ 完整的4步thinking过程（精简版）
- ✅ 最终输出的所有热点
- ✅ JSON格式，直接可用

#### 紧凑版 (~170 tokens)
- **`output/few_shot_compact_上.txt`** (上方向)
- **`output/few_shot_compact_下.txt`** (下方向)

包含内容：
- ✅ 输入数量和方向
- ✅ 聚类识别结果（前3个cluster）
- ✅ 输出热点摘要

**推荐使用紧凑版**，因为token消耗更少。

## 使用方法

### 方法1: 添加到System Prompt（推荐）

编辑 `prompt_templates.py` 中的 `get_system_prompt()` 函数：

```python
@staticmethod
def get_system_prompt() -> str:
    """获取系统prompt"""

    # 读取few-shot示例
    import os
    few_shot_path = os.path.join(os.path.dirname(__file__),
                                  'output/few_shot_compact_上.txt')
    if os.path.exists(few_shot_path):
        with open(few_shot_path, 'r', encoding='utf-8') as f:
            few_shot_example = f.read()
    else:
        few_shot_example = ""

    return f"""你是资深交通拥堵预测专家，擅长基于历史数据分析预测未来拥堵区域。

请使用结构化思维链方式分析数据：
1. 先进行空间-时间聚类
2. 分析每个聚类的几何边界
3. 计算预测形状的顶点坐标
4. 验证结果的合理性

{few_shot_example}

输出必须是严格的JSON格式，包含thinking字段记录你的推理过程。不要添加任何markdown标记。"""
```

### 方法2: 添加到User Prompt开头

编辑 `get_hotspot_analysis_prompt()` 函数，在开头添加：

```python
# 在prompt开始处添加
prompt = f"""
<few_shot_reference>
参考以下成功案例的分析方式：

{few_shot_example_content}
</few_shot_reference>

<task>
分析过去三年同一日期的交通拥堵数据，预测今年同一日期可能发生拥堵的区域。
</task>
...
"""
```

### 方法3: 创建专门的Few-Shot Section

在 `get_hotspot_analysis_prompt()` 的 `<methodology>` 部分之前插入：

```python
prompt = f"""
...

<few_shot_examples>
{few_shot_example_content}
</few_shot_examples>

<methodology>
...
"""
```

## 生成新的Few-Shot示例

### 步骤1: 运行工作流得到新的结果

```bash
# 运行你的工作流
python test_complete_workflow.py

# 这会生成新的 output/workflow_state.json
```

### 步骤2: 提取Few-Shot示例

```bash
# 提取完整数据
python extract_few_shot_examples.py --readable

# 生成prompt格式（上方向）
python create_few_shot_prompt.py --direction 上 --compact

# 生成prompt格式（下方向）
python create_few_shot_prompt.py --direction 下 --compact
```

### 步骤3: 选择最佳示例

查看生成的文件，选择聚类效果最好的示例：

```bash
# 查看上方向示例
cat output/few_shot_compact_上.txt

# 查看下方向示例
cat output/few_shot_compact_下.txt
```

**选择标准**：
- ✓ 聚类数量合理（3-5个）
- ✓ 覆盖率高（>70%）
- ✓ 热点边界清晰
- ✓ 没有明显的过度合并或过度拆分

### 步骤4: 添加到Prompt

将选中的示例内容复制到 `prompt_templates.py` 中。

## 示例对比

### 紧凑版示例（推荐）

```
<few_shot_example_compact>

输入: 35个三角形 (上方向)

聚类识别:
  Cluster 1包含三角形[0, 19]：- KP跨度: 0.0-4.2km（4.2km）- 时间跨度: 16:20-19:20（180分钟）- 覆盖年份：2014、2018、2024三年。
  Cluster 2包含三角形[1, 6, 10]：- KP跨度: 31.4-68.3km（36.9km）- 时间跨度: 18:00-22:00（240分钟）- 覆盖年份：2014、2018、2024三年。
  Cluster 3包含三角形[2, 3, 5, 7]：- KP跨度: 33.0-49.5km（16.5km）- 时间跨度: 15:25-18:15（170分钟）- 覆盖年份：2014、2018、2024三年。

输出: 5个热点
  Hotspot 1: KP 0.0-4.2km, 16:20-19:20, 2个事件, high
  Hotspot 2: KP 31.4-68.3km, 18:00-22:00, 3个事件, high
  Hotspot 3: KP 33.0-49.5km, 15:25-18:15, 4个事件, high
  Hotspot 4: KP 55.1-109.3km, 14:55-19:25, 3个事件, high
  Hotspot 5: KP 93.8-111.2km, 17:05-19:50, 3个事件, high

</few_shot_example_compact>
```

**Token消耗**: ~170 tokens

### 完整版示例（可选）

包含完整的输入数据、4步thinking过程、详细的输出结果。

**Token消耗**: ~2200 tokens

## Few-Shot的优势

1. **提供具体范例**：让LLM理解期望的聚类粒度
2. **统一输出格式**：确保输出格式一致
3. **提高聚类质量**：通过好的示例引导LLM
4. **减少试错**：避免LLM产生不合理的聚类

## 注意事项

### Token预算

Few-shot示例会消耗额外的tokens：
- 紧凑版：~170 tokens
- 完整版：~2200 tokens

如果遇到token限制，建议：
1. 使用紧凑版
2. 只展示1-2个方向的示例
3. 只保留最关键的聚类信息

### 示例选择

选择few-shot示例时应考虑：
- ✓ 与当前任务相似的数据规模
- ✓ 聚类效果优秀的案例
- ✓ 包含多种聚类场景（大聚类、小聚类）

### 动态加载

可以根据输入数据动态选择示例：

```python
def select_few_shot_example(triangle_count: int):
    """根据三角形数量选择合适的示例"""
    if triangle_count < 20:
        return load_example('few_shot_small.txt')
    elif triangle_count < 50:
        return load_example('few_shot_medium.txt')
    else:
        return load_example('few_shot_large.txt')
```

## 验证Few-Shot效果

添加few-shot后，验证效果：

1. **运行测试**
```bash
python test_complete_workflow.py
```

2. **查看thinking链**
```bash
python view_llm_thinking.py --summary
```

3. **对比指标**
- 聚类数量是否合理
- 覆盖率是否提高
- 聚类边界是否清晰
- 是否减少了过度合并/拆分

## 多示例策略

可以提供多个示例，展示不同场景：

```xml
<few_shot_examples>

<example_1>
场景：少量三角形（~20个）
...
</example_1>

<example_2>
场景：中等数量三角形（~40个）
...
</example_2>

<example_3>
场景：大量三角形（~80个）
...
</example_3>

</few_shot_examples>
```

## 生成的文件列表

```
output/
├── few_shot_example.json              # 完整结构化数据 (67KB)
├── few_shot_example_readable.txt      # 人类可读格式 (8.2KB)
├── few_shot_prompt_上.txt             # 上方向完整prompt (~9KB)
├── few_shot_prompt_下.txt             # 下方向完整prompt (~9KB)
├── few_shot_compact_上.txt            # 上方向紧凑版 (~700B)
└── few_shot_compact_下.txt            # 下方向紧凑版 (~700B)
```

## 快速开始

最简单的使用方式：

```bash
# 1. 提取few-shot示例
python create_few_shot_prompt.py --compact

# 2. 查看紧凑版内容
cat output/few_shot_compact_上.txt

# 3. 复制内容到prompt_templates.py的system prompt中

# 4. 测试效果
python test_complete_workflow.py
python view_llm_thinking.py --summary
```

## 常见问题

### Q: 应该使用哪个方向的示例？

**A**: 建议两个方向都添加，或者选择数据质量更好的方向。

### Q: 紧凑版和完整版如何选择？

**A**:
- 如果token充足，用完整版（包含详细thinking）
- 如果token紧张，用紧凑版（只包含关键结果）
- 推荐从紧凑版开始测试

### Q: Few-shot会影响新数据的分析吗？

**A**: 不会。Few-shot只是提供参考，LLM仍会根据实际输入数据进行分析。它主要影响：
- 聚类的粒度偏好
- 输出格式的一致性
- 边界处理的方式

### Q: 如何更新Few-shot示例？

**A**:
1. 运行新的工作流得到更好的结果
2. 重新运行提取脚本
3. 替换prompt中的示例内容
4. 测试验证

### Q: 可以手动编辑Few-shot示例吗？

**A**: 可以！你可以：
- 修改聚类边界使其更合理
- 调整热点数量
- 简化或扩展描述
- 但要保持JSON格式正确

## 总结

Few-shot learning通过提供具体示例，帮助LLM更好地理解任务要求。按照以下步骤使用：

1. ✅ 运行工作流生成结果
2. ✅ 提取few-shot示例
3. ✅ 选择最佳示例
4. ✅ 添加到prompt
5. ✅ 测试验证效果

**推荐配置**：使用紧凑版示例添加到system prompt中，token消耗仅~170，但能显著提高聚类质量。
