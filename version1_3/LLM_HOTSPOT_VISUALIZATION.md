# 🎨 LLM热点可视化与对比功能

## 📋 新增功能概述

现在系统已经完全集成了LLM热点识别和可视化对比功能：

1. ✅ **LLM生成三角形坐标** - LLM在识别热点时会生成外包大三角形的坐标
2. ✅ **对比可视化** - 自动生成原始三角形 vs LLM热点的对比图
3. ✅ **CSV输出支持** - CSV生成器支持LLM分析结果
4. ✅ **方向感知** - 按上行/下行方向分别处理和可视化

## 🔄 完整工作流

```
用户输入
   ↓
chatbot (解析)
   ↓
visualization (生成原始三角形图)
   ↓
analyze_with_llm (LLM分析 + 生成对比图)
   ↓
report (生成报告 + CSV输出)
```

## 📊 输出文件

运行后会在 `output/` 目录生成：

### 1️⃣ 原始三角形可视化
```
triangles_multi_関越道_上り_YYYYMMDD_HHMMSS.png
triangles_multi_関越道_下り_YYYYMMDD_HHMMSS.png
```
- 显示所有原始拥堵事件的三角形

### 2️⃣ 对比可视化图（新增⭐）
```
comparison_関越道_上り_YYYYMMDD_HHMMSS.png
comparison_関越道_下り_YYYYMMDD_HHMMSS.png
```
- **左图**: 原始拥堵事件三角形（浅蓝色）
- **右图**:
  - 背景：原始三角形（浅灰色）
  - 前景：LLM识别的热点区域（彩色大三角形）
  - 颜色编码：
    - 🟢 绿色：low severity
    - 🟡 黄色：medium severity
    - 🔴 红色：high severity

### 3️⃣ CSV预测输出
```
congestion_prediction_関越道_YYYYMMDD_HHMMSS.csv
```
- 基于LLM分析结果生成的预测数据

### 4️⃣ 工作流状态
```
workflow_state.json
```
- 完整的工作流执行状态

## 🎯 LLM输出格式

LLM现在会为每个热点返回：

```json
{
    "hotspot_id": 1,
    "kp_range": [23.5, 31.2],
    "time_range": [960, 1200],
    "included_triangle_ids": [1, 3, 5, 8],
    "frequency": 5,
    "severity": "high",
    "description": "晚高峰拥堵热点",
    "hull_triangle": {
        "kp_start": 23.5,
        "kp_end": 31.2,
        "peak_kp": 27.3,
        "time_start": 960,
        "time_end": 1200,
        "peak_time": 1080
    }
}
```

**hull_triangle** 字段包含外包大三角形的6个坐标：
- `kp_start`, `kp_end`: 底边的起止KP
- `peak_kp`: 顶点的KP位置
- `time_start`, `time_end`: 底边的起止时间
- `peak_time`: 顶点的时间

## 🔍 可视化对比说明

### 左图：原始拥堵事件
- 每个小三角形代表一次拥堵事件
- 浅蓝色填充，蓝色边框
- 显示所有检测到的拥堵

### 右图：LLM识别的热点
- **背景层**：原始三角形（灰色，半透明）
- **热点层**：LLM识别的热点区域大三角形
  - 彩色填充（根据严重程度）
  - 深红色粗边框
  - 中心标注：热点编号 + 拥堵频次
- **图例**：显示每个热点的严重程度

### 方向处理
- 系统会自动识别上行/下行方向
- 为每个方向生成独立的对比图
- 确保三角形方向正确（根据KP值方向）

## 💻 使用方法

### 方法1: 使用测试脚本

```bash
python test_complete_workflow.py
```

选择交互模式，输入查询，系统会自动：
1. 生成原始三角形图
2. LLM分析热点
3. 生成对比可视化图
4. 输出CSV预测

### 方法2: 直接调用API

```python
from implementation import compiled_agent

result = compiled_agent.invoke({
    "user_input": "请分析関越高速公路2024年5月3日的交通情况"
})

# 查看生成的文件
print("原始三角形图:", result['fig_paths'])
print("对比图:", result['comparison_fig_paths'])
print("CSV输出:", result['csv_output_path'])
```

### 方法3: 自定义可视化

```python
from visualization_comparison import visualize_comparison, convert_hotspot_to_hull

# 准备数据
triangles = [...]  # 原始三角形
hotspots = [...]   # LLM识别的热点

# 转换hotspot为hull格式
llm_hulls = [convert_hotspot_to_hull(h, "上") for h in hotspots]

# 生成对比图
visualize_comparison(
    triangles=triangles,
    llm_hulls=llm_hulls,
    direction="上",
    output_path="output/my_comparison.png"
)
```

## 🔧 自定义配置

### 修改可视化样式

编辑 `visualization_comparison.py`：

```python
# 修改颜色方案
severity_colors = {
    'low': '#90EE90',    # 浅绿色
    'medium': '#FFD700',  # 金色
    'high': '#FF6B6B'     # 红色
}

# 修改图片大小
figsize = (20, 14)  # 更大的图片

# 修改透明度
alpha=0.6  # 更不透明
```

### 修改LLM生成的三角形

编辑 `prompt_templates.py` 中的prompt，调整：
- 如何计算峰值位置
- 如何确定三角形范围
- 其他几何特征

## 🎨 可视化特点

### 1. 双视图对比
- 左右并排显示
- 相同的坐标轴范围
- 便于直观对比

### 2. 图层叠加
- 原始数据作为背景
- LLM热点作为前景
- 清晰显示聚集关系

### 3. 信息丰富
- 统计信息框
- 热点编号标注
- 频次和严重程度显示
- 图例说明

### 4. 高质量输出
- 150 DPI分辨率
- 自动调整布局
- 中文字体支持

## 📈 数据流程

```
原始CSV数据
    ↓
生成三角形坐标
    ↓
传给LLM分析
    ↓
LLM返回热点 + hull_triangle坐标
    ↓
┌─────────────────────┬──────────────────────┐
│ 可视化对比          │ CSV输出              │
│ (左图vs右图)        │ (hotspot转hull格式)  │
└─────────────────────┴──────────────────────┘
```

## ⚙️ 技术细节

### hotspot → hull 转换

`convert_hotspot_to_hull()` 函数将LLM的hotspot转换为传统的hull格式：

```python
hull = {
    'cluster_id': hotspot['hotspot_id'],
    'cluster_size': hotspot['frequency'],
    'vertices': [[kp_start, time_start], [kp_end, time_end], [peak_kp, peak_time]],
    'kp_range': [kp_start, kp_end],
    'time_range': [time_start, time_end],
    'direction': direction,
    'severity': hotspot['severity'],
    'source': 'llm'
}
```

### 方向处理

系统会根据KP值的增减方向自动判断：
- 上行：KP值递增
- 下行：KP值递减

三角形的绘制会根据方向调整。

## 🐛 故障排除

### Q1: 对比图没有生成

**检查**：
- LLM是否成功返回hotspot数据
- hotspot是否包含hull_triangle字段
- 查看控制台是否有错误信息

**解决**：检查prompt template是否正确要求LLM返回hull_triangle

### Q2: CSV文件没有生成

**原因**：LLM分析结果没有被正确转换

**解决**：
- 确认 `llm_analysis` 在state中存在
- 检查 `excel_output_generator.py` 中的转换逻辑

### Q3: 三角形方向不对

**检查**：
- direction参数是否正确传递
- KP值是否按预期递增/递减

**解决**：在 `convert_hotspot_to_hull()` 中添加方向校验逻辑

## 📚 相关文件

| 文件 | 功能 |
|------|------|
| `prompt_templates.py` | LLM prompt定义，包含hull_triangle要求 |
| `visualization_comparison.py` | 对比可视化核心代码 |
| `implementation.py` | analyze_with_llm节点，调用可视化 |
| `excel_output_generator.py` | CSV生成，支持LLM hotspot |
| `llm_analyzer.py` | LLM API调用 |

## 🎓 最佳实践

1. **先检查原始三角形图** - 确认数据质量
2. **查看对比图** - 评估LLM识别效果
3. **检查CSV输出** - 确认预测数据正确
4. **调整prompt** - 如果识别不理想，修改prompt
5. **保存重要结果** - 对比图可以用于报告和分析

## 🚀 后续增强

可能的改进方向：
1. 添加交互式可视化（使用Plotly）
2. 支持多模态输入（传入图片给LLM）
3. 生成GIF动画展示时间演变
4. 添加3D可视化（KP × 时间 × 拥堵强度）
5. 支持热力图叠加

---

*更新时间: 2025-09-30*
*版本: v1.3.1*