# 🤖 analyze_with_llm 节点详解

## 📋 目录

1. [节点概述](#节点概述)
2. [完整执行流程](#完整执行流程)
3. [关键组件详解](#关键组件详解)
4. [数据流转](#数据流转)
5. [LLM Prompt构建](#llm-prompt构建)
6. [输出结果](#输出结果)
7. [错误处理](#错误处理)

---

## 节点概述

### 🎯 功能
使用大语言模型（LLM）智能识别交通拥堵热点区域，替代传统的DBSCAN聚类算法。

### 📍 在工作流中的位置
```
chatbot → visualization → analyze_with_llm → report
                              ↑
                            在这里
```

### 📥 输入数据
从 `state` 接收：
```python
{
    "direction_data": {
        "上": {
            "triangles": [
                {
                    "id": 0,
                    "kp_start": 10.0,
                    "kp_end": 15.0,
                    "peak_kp": 12.5,
                    "time_start": 480,
                    "time_end": 600,
                    "peak_time": 540
                },
                # ... 更多三角形
            ],
            "fig_path": "output/triangles_*.png"
        },
        "下": { ... }
    },
    "file_paths": ["関越道_上_2024_05-03.csv", ...],
    "route": "関越道"
}
```

### 📤 输出数据
返回给 `state`：
```python
{
    "llm_analysis": {
        "上": {
            "direction": "上",
            "hotspots": [
                {
                    "hotspot_id": 1,
                    "kp_range": [23.5, 31.2],
                    "time_range": [960, 1200],
                    "frequency": 5,
                    "severity": "high",
                    "hull_triangle": { ... }
                }
            ],
            "summary": {
                "total_hotspots": 3,
                "analysis_confidence": 0.85
            }
        },
        "下": { ... }
    },
    "comparison_fig_paths": [
        "output/comparison_関越道_上り_*.png",
        "output/comparison_関越道_下り_*.png"
    ]
}
```

---

## 完整执行流程

### 第一步：数据准备 (implementation.py:630-642)

```python
def analyze_with_llm(state: TrafficState) -> dict:
    # 1. 从state获取输入数据
    direction_data = state.get("direction_data", {})
    file_paths = state.get("file_paths", [])

    # 2. 提取CSV文件名
    csv_files = [os.path.basename(fp) for fp in file_paths]
```

**这一步做什么？**
- 获取前一个节点（visualization）生成的三角形数据
- `direction_data` 包含按方向分组的所有三角形
- 提取CSV文件名用于告诉LLM数据来源

**数据示例：**
```python
direction_data = {
    "上": {
        "triangles": [6个三角形],
        "fig_path": "output/triangles_上り.png"
    },
    "下": {
        "triangles": [58个三角形],
        "fig_path": "output/triangles_下り.png"
    }
}

csv_files = [
    "関越道_上_2024_05-03.csv",
    "関越道_下_2024_05-03.csv"
]
```

---

### 第二步：创建LLM分析器 (implementation.py:645-655)

```python
# 3. 检查API密钥
api_key = config.get_openai_api_key()

# 4. 创建批量分析器
batch_analyzer = create_batch_analyzer(
    api_key=api_key,
    model=config.openai_model  # 默认 "gpt-4o-mini"
)
```

**这一步做什么？**
- 检查是否配置了OpenAI API密钥
- 创建一个批量分析器，可以同时处理多个方向

**深入：`create_batch_analyzer` 在做什么？**

查看 `llm_analyzer.py:258-269`：

```python
def create_batch_analyzer(api_key: str, model: str = "gpt-4o-mini"):
    # 创建单个LLM分析器
    llm_analyzer = LLMAnalyzer(
        api_key=api_key,
        model=model,
        temperature=0.1,      # 低温度 = 更稳定的输出
        max_tokens=4000,      # 最多4000个token
        timeout=30,           # 30秒超时
        use_fallback=True     # 如果失败，使用fallback
    )

    # 包装成批量分析器
    return BatchLLMAnalyzer(llm_analyzer)
```

---

### 第三步：批量分析所有方向 (implementation.py:657-661)

```python
# 5. 分析所有方向（上、下）
llm_analysis = batch_analyzer.analyze_all_directions(
    direction_data=direction_data,
    csv_files=csv_files
)
```

**这一步做什么？**
- 对每个方向（上、下）分别调用LLM
- 返回每个方向的热点分析结果

**深入：`analyze_all_directions` 在做什么？**

查看 `llm_analyzer.py:227-256`：

```python
def analyze_all_directions(self, direction_data, csv_files):
    results = {}

    # 遍历每个方向
    for direction, data in direction_data.items():
        triangles = data.get('triangles', [])

        print(f"Analyzing {direction} direction...")

        # 为这个方向调用LLM分析
        result = self.llm_analyzer.analyze_hotspots(
            triangles=triangles,
            direction=direction,
            csv_files=csv_files
        )

        results[direction] = result

    return results
```

**执行示例：**
```
Input: direction_data = {"上": {...}, "下": {...}}

处理:
  第1次循环: direction="上", triangles=[6个]
    → 调用LLM分析 → result_上
  第2次循环: direction="下", triangles=[58个]
    → 调用LLM分析 → result_下

Output: {"上": result_上, "下": result_下}
```

---

### 第四步：单个方向的LLM分析 (核心！)

查看 `llm_analyzer.py:42-90`：

```python
def analyze_hotspots(self, triangles, direction, csv_files):
    """分析单个方向的热点"""

    # 4.1 调用LLM API
    result = self._call_llm_api(triangles, direction, csv_files)

    # 4.2 验证结果格式
    if self._validate_result(result):
        return result  # 成功
    else:
        return self._use_fallback(triangles, direction)  # 失败，用fallback
```

#### 4.1 调用LLM API (llm_analyzer.py:92-128)

```python
def _call_llm_api(self, triangles, direction, csv_files):
    # Step 1: 构建prompt
    user_prompt = build_analysis_prompt(
        direction=direction,
        csv_files=csv_files,
        triangles=triangles
    )

    system_prompt = "你是交通拥堵分析专家，必须严格返回JSON格式"

    # Step 2: 调用OpenAI API
    response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=4000,
        response_format={"type": "json_object"}  # ⭐ 强制JSON输出
    )

    # Step 3: 解析JSON
    result_text = response.choices[0].message.content
    result = json.loads(result_text)

    return result
```

**关键点：**
- `response_format={"type": "json_object"}` 强制LLM返回JSON
- 温度0.1保证输出稳定
- 解析成Python字典返回

---

### 第五步：Prompt构建 (重要！)

查看 `prompt_templates.py:44-150`：

#### Prompt结构

```
你是交通分析专家...

## 数据概览
- 方向: 上
- 拥堵事件数量: 6
- KP范围: 0.0 - 50.0 km
- 时间范围: 00:00 - 24:00

## 拥堵事件详情
[
    {
        "id": 0,
        "kp_start": 10.0,
        "kp_end": 15.0,
        "peak_kp": 12.5,
        "time_start": 480,
        "time_end": 600,
        "peak_time": 540
    },
    ...
]

## 分析任务
识别在时间和空间上聚集的拥堵热点...

判断标准：
1. 空间聚集性: 5-10km范围
2. 时间聚集性: 2-4小时范围
3. 最少事件数: 2-3个

## 输出格式
必须严格按照以下JSON格式：
{
    "direction": "上",
    "hotspots": [
        {
            "hotspot_id": 1,
            "kp_range": [23.5, 31.2],
            "time_range": [960, 1200],
            "included_triangle_ids": [1, 3, 5],
            "frequency": 5,
            "severity": "high",
            "description": "...",
            "hull_triangle": {
                "kp_start": 23.5,
                "kp_end": 31.2,
                "peak_kp": 27.3,
                "time_start": 960,
                "time_end": 1200,
                "peak_time": 1080
            }
        }
    ],
    "summary": {
        "total_hotspots": 1,
        "most_severe_hotspot_id": 1,
        "analysis_confidence": 0.85
    }
}
```

**Prompt如何构建？**

```python
def build_analysis_prompt(direction, csv_files, triangles):
    # 1. 提取统计信息
    kp_values = [t['kp_start'], t['kp_end'] for t in triangles]
    time_values = [t['time_start'], t['time_end'] for t in triangles]

    kp_range = [min(kp_values), max(kp_values)]
    time_range = [min(time_values), max(time_values)]

    # 2. 准备三角形数据（标准化格式）
    triangles_data = [
        {
            "id": i,
            "kp_start": round(t['kp_start'], 2),
            "kp_end": round(t['kp_end'], 2),
            "peak_kp": round(t['peak_kp'], 2),
            "time_start": int(t['time_start']),
            "time_end": int(t['time_end']),
            "peak_time": int(t['peak_time'])
        }
        for i, t in enumerate(triangles)
    ]

    # 3. 填充到模板
    return PromptTemplates.get_hotspot_analysis_prompt(
        direction=direction,
        csv_files=csv_files,
        triangle_count=len(triangles),
        kp_range=kp_range,
        time_range=time_range,
        triangles_data=triangles_data
    )
```

---

### 第六步：结果验证 (llm_analyzer.py:130-158)

```python
def _validate_result(self, result):
    """验证LLM返回格式是否正确"""
    try:
        # 检查必需字段
        if 'direction' not in result:
            return False
        if 'hotspots' not in result:
            return False
        if 'summary' not in result:
            return False

        # 检查每个hotspot的格式
        for hotspot in result['hotspots']:
            required_fields = [
                'hotspot_id',
                'kp_range',
                'time_range',
                'frequency',
                'severity',
                'hull_triangle'  # ⭐ 必须有三角形坐标
            ]

            if not all(field in hotspot for field in required_fields):
                return False

        return True

    except:
        return False
```

**如果验证失败？**
→ 使用fallback分析器（基于规则的简单聚类）

---

### 第七步：生成对比可视化 (implementation.py:678-694)

```python
# 6. 生成对比可视化图
comparison_paths = visualize_all_directions_comparison(
    direction_data=direction_data,
    llm_analysis=llm_analysis,
    output_dir="output",
    road_name=state.get('route', '関越道')
)
```

**这一步做什么？**
- 为每个方向生成对比图
- 左图：原始三角形
- 右图：LLM识别的热点

**深入：对比图生成流程**

```python
def visualize_all_directions_comparison(direction_data, llm_analysis, ...):
    comparison_paths = {}

    for direction, data in direction_data.items():
        triangles = data['triangles']  # 原始三角形

        # 获取LLM的hotspots
        hotspots = llm_analysis[direction]['hotspots']

        # 转换hotspot为hull格式（用于绘制）
        llm_hulls = [
            convert_hotspot_to_hull(h, direction)
            for h in hotspots
        ]

        # 生成对比图
        output_path = f"output/comparison_{direction}ri_*.png"

        visualize_comparison(
            triangles=triangles,      # 左图
            llm_hulls=llm_hulls,      # 右图
            direction=direction,
            output_path=output_path
        )

        comparison_paths[direction] = output_path

    return comparison_paths
```

---

### 第八步：返回结果 (implementation.py:696-699)

```python
return {
    "llm_analysis": llm_analysis,
    "comparison_fig_paths": list(comparison_paths.values())
}
```

这些数据会合并到 `state`，供下一个节点（report）使用。

---

## 关键组件详解

### 1. LLMAnalyzer 类

**位置**: `llm_analyzer.py:22-204`

**作用**: 封装LLM API调用逻辑

**主要方法**:
```python
class LLMAnalyzer:
    def __init__(api_key, model, temperature, ...):
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key)
        self.fallback_analyzer = FallbackAnalyzer()

    def analyze_hotspots(triangles, direction, csv_files):
        # 主要分析方法
        result = self._call_llm_api(...)
        if self._validate_result(result):
            return result
        else:
            return self._use_fallback(...)

    def _call_llm_api(...):
        # 调用OpenAI API

    def _validate_result(...):
        # 验证JSON格式

    def _use_fallback(...):
        # 降级到规则分析
```

---

### 2. BatchLLMAnalyzer 类

**位置**: `llm_analyzer.py:207-256`

**作用**: 批量处理多个方向

**主要方法**:
```python
class BatchLLMAnalyzer:
    def __init__(llm_analyzer):
        self.llm_analyzer = llm_analyzer

    def analyze_all_directions(direction_data, csv_files):
        results = {}
        for direction, data in direction_data.items():
            result = self.llm_analyzer.analyze_hotspots(...)
            results[direction] = result
        return results
```

---

### 3. FallbackAnalyzer 类

**位置**: `fallback_analyzer.py:13-162`

**作用**: 当LLM不可用时的备用方案

**算法**: 基于距离的简单聚类

```python
class FallbackAnalyzer:
    def __init__(kp_threshold=5.0, time_threshold=180):
        # KP阈值5km，时间阈值180分钟

    def analyze(triangles, direction):
        # 1. 计算距离矩阵
        # 2. 贪心聚类
        # 3. 生成hotspots（格式与LLM一致）
        return {
            "direction": direction,
            "hotspots": [...],
            "summary": {...}
        }
```

---

## 数据流转

### 完整的数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. visualization节点输出                                      │
│    direction_data = {                                        │
│        "上": {"triangles": [6个], "fig_path": "..."},       │
│        "下": {"triangles": [58个], "fig_path": "..."}       │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. analyze_with_llm节点接收                                  │
│    从state获取direction_data和file_paths                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 为每个方向准备数据                                         │
│    上: triangles=[6个] → prepare → llm_input_上              │
│    下: triangles=[58个] → prepare → llm_input_下             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 构建Prompt并调用LLM                                       │
│    llm_input_上 → build_prompt → LLM API → result_上        │
│    llm_input_下 → build_prompt → LLM API → result_下        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 验证和处理结果                                             │
│    result_上 → validate → hotspots_上                        │
│    result_下 → validate → hotspots_下                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 生成对比可视化                                             │
│    triangles_上 + hotspots_上 → comparison_上.png           │
│    triangles_下 + hotspots_下 → comparison_下.png           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 返回结果到state                                           │
│    {                                                         │
│        "llm_analysis": {上: {...}, 下: {...}},              │
│        "comparison_fig_paths": [path1, path2]               │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. report节点使用                                            │
│    llm_analysis → 生成报告 + CSV输出                         │
└─────────────────────────────────────────────────────────────┘
```

---

## LLM Prompt构建

### Prompt的三个部分

#### 1. 系统Prompt (System Message)
```
你是一个专业的交通拥堵分析专家。
你必须严格按照JSON格式返回分析结果。
```

**作用**: 设定LLM的角色和行为规范

#### 2. 用户Prompt (User Message)

**结构**:
```
1. 数据概览
   - 方向
   - 拥堵事件数量
   - KP范围
   - 时间范围

2. 拥堵事件详情
   - 完整的三角形数据JSON

3. 分析任务
   - 识别热点的标准
   - 判断聚集性的方法

4. 输出格式
   - JSON schema
   - 示例输出
```

**作用**: 提供具体的分析任务和数据

#### 3. 输出格式约束

```python
response_format={"type": "json_object"}
```

**作用**: 强制LLM返回有效JSON

---

### Prompt示例（实际发送给LLM的）

```
你是一个专业的交通拥堵分析专家。请分析以下上方向的拥堵数据...

## 数据概览
- 方向: 上
- 数据源文件: 関越道_上_2024_05-03.csv
- 拥堵事件数量: 6
- KP范围: 0.0 - 8.5 km
- 时间范围: 06:25 - 09:50

## 拥堵事件详情
[
  {
    "id": 0,
    "kp_start": 0.0,
    "kp_end": 0.5,
    "peak_kp": 0.2,
    "time_start": 385,
    "time_start": 455,
    "peak_time": 420
  },
  {
    "id": 1,
    "kp_start": 0.8,
    "kp_end": 1.2,
    "peak_kp": 1.0,
    "time_start": 475,
    "time_end": 590,
    "peak_time": 532
  },
  ...
]

## 分析任务
请识别在时间和空间上聚集的拥堵热点区域...

判断标准：
1. 空间聚集性: 多个事件在5-10km范围内
2. 时间聚集性: 在2-4小时范围内
3. 频次要求: 至少2-3个事件

## 输出格式要求
必须严格按照以下JSON格式输出...
```

---

## 输出结果

### LLM返回的原始JSON

```json
{
    "direction": "上",
    "hotspots": [
        {
            "hotspot_id": 1,
            "kp_range": [0.0, 1.2],
            "time_range": [385, 590],
            "included_triangle_ids": [0, 1],
            "frequency": 2,
            "severity": "low",
            "description": "早晨6点到10点，KP0-1.2区间拥堵",
            "hull_triangle": {
                "kp_start": 0.0,
                "kp_end": 1.2,
                "peak_kp": 0.6,
                "time_start": 385,
                "time_end": 590,
                "peak_time": 487
            }
        }
    ],
    "summary": {
        "total_hotspots": 1,
        "most_severe_hotspot_id": 1,
        "analysis_confidence": 0.90
    }
}
```

### 存储到state的格式

```python
state["llm_analysis"] = {
    "上": {
        "direction": "上",
        "hotspots": [...],
        "summary": {...}
    },
    "下": {
        "direction": "下",
        "hotspots": [...],
        "summary": {...}
    }
}

state["comparison_fig_paths"] = [
    "output/comparison_関越道_上り_20250503_233901.png",
    "output/comparison_関越道_下り_20250503_233901.png"
]
```

---

## 错误处理

### 1. 没有API密钥

```python
if not api_key:
    print("⚠️ No API key, using fallback")
    return fallback_analyzer.analyze(triangles, direction)
```

→ 自动使用fallback分析器

### 2. LLM API调用失败

```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"❌ API call failed: {e}")
    return self._use_fallback(triangles, direction)
```

→ 捕获异常，使用fallback

### 3. JSON格式验证失败

```python
if not self._validate_result(result):
    print("⚠️ Invalid format")
    return self._use_fallback(triangles, direction)
```

→ 格式错误，使用fallback

### 4. 可视化生成失败

```python
try:
    comparison_paths = visualize_all_directions_comparison(...)
except Exception as e:
    print(f"⚠️ Visualization failed: {e}")
    comparison_paths = {}
```

→ 继续执行，只是没有对比图

---

## 调试技巧

### 1. 查看LLM输入

在 `llm_analyzer.py:92` 添加：

```python
def _call_llm_api(self, triangles, direction, csv_files):
    user_prompt = build_analysis_prompt(...)

    # 调试：打印prompt
    print("="*70)
    print("LLM Input Prompt:")
    print(user_prompt[:500])  # 前500字符
    print("="*70)

    response = self.client.chat.completions.create(...)
```

### 2. 查看LLM输出

在 `llm_analyzer.py:118` 添加：

```python
result_text = response.choices[0].message.content

# 调试：打印原始输出
print("="*70)
print("LLM Raw Output:")
print(result_text)
print("="*70)

result = json.loads(result_text)
```

### 3. 强制使用Fallback测试

```python
# 在implementation.py中临时修改
api_key = None  # 强制使用fallback
```

### 4. 查看中间数据

```python
# 在analyze_with_llm中添加
print("Direction Data:")
for direction, data in direction_data.items():
    print(f"  {direction}: {len(data['triangles'])} triangles")
    print(f"    First triangle: {data['triangles'][0]}")
```

---

## 总结

### 核心流程
```
输入三角形坐标
    ↓
构建详细Prompt
    ↓
调用OpenAI API
    ↓
强制返回JSON
    ↓
验证格式
    ↓
生成对比可视化
    ↓
返回结果
```

### 关键特性
- ✅ 智能识别聚集模式
- ✅ 自动生成三角形坐标
- ✅ 严格JSON格式输出
- ✅ 完善的错误处理
- ✅ Fallback机制
- ✅ 方向感知处理
- ✅ 自动可视化对比

### 优势
- 比DBSCAN更灵活
- 能理解复杂模式
- 自然语言描述
- 可调整分析标准（修改prompt）

---

*希望这个详解能帮你完全理解 analyze_with_llm 节点的运作！*