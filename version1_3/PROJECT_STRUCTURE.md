# 项目结构与依赖关系

## 📁 项目概览

这是一个基于 LangGraph 的交通拥堵热点分析和预测系统，使用 LLM 进行智能分析。

---

## 🎯 主要入口脚本（按重要性排序）

### 1. **test_complete_workflow.py** (432行) ⭐⭐⭐⭐⭐
**用途**: 完整工作流测试，端到端演示整个系统
**依赖**:
```
├── implementation.py (chatbot, visualization, analyze_with_llm, report)
└── config.py
```
**运行方式**:
```bash
python test_complete_workflow.py
```
**功能**: 测试从用户输入到最终报告的完整流程

---

### 2. **batch_evaluation.py** (439行) ⭐⭐⭐⭐⭐
**用途**: 批量评估脚本，自动运行多个预测任务并评估
**依赖**:
```
├── evaluation.py (Evaluator)
├── implementation.py (compiled_agent)
└── functions.py (process_direction_aware_traffic_data)
```
**运行方式**:
```bash
python batch_evaluation.py --config evaluation_config.json
```
**功能**:
- 读取配置文件
- 批量运行预测
- 自动评估结果
- 生成汇总报告

---

### 3. **test_llm_workflow.py** (约100行) ⭐⭐⭐⭐
**用途**: 测试 LLM 分析工作流
**依赖**:
```
├── implementation.py (compiled_agent)
└── config.py
```
**运行方式**:
```bash
python test_llm_workflow.py
```
**功能**: 快速测试 LLM 分析功能是否正常

---

### 4. **test_evaluation.py** (273行) ⭐⭐⭐⭐
**用途**: 测试评估系统
**依赖**:
```
├── evaluation.py (Evaluator)
└── functions.py (process_direction_aware_traffic_data)
```
**运行方式**:
```bash
python test_evaluation.py --mode sample  # 使用示例数据
python test_evaluation.py --mode real    # 使用真实数据
```
**功能**: 测试三种 IoU 评估指标

---

### 5. **view_llm_thinking.py** (231行) ⭐⭐⭐
**用途**: 查看 LLM 推理过程
**依赖**: 无（独立工具）
**运行方式**:
```bash
python view_llm_thinking.py [workflow_state.json路径]
```
**功能**: 从 workflow_state.json 提取并美化显示 LLM 的思考链

---

### 6. **create_few_shot_prompt.py** (283行) ⭐⭐
**用途**: 创建少样本学习提示词
**依赖**:
```
└── prompt_templates.py
```
**运行方式**:
```bash
python create_few_shot_prompt.py
```
**功能**: 生成用于 LLM 的 few-shot 示例

---

### 7. **extract_few_shot_examples.py** (332行) ⭐⭐
**用途**: 从数据中提取少样本示例
**依赖**:
```
└── functions.py
```
**功能**: 从历史数据中提取高质量的热点识别示例

---

## 🔧 核心模块（被引用的脚本）

### 1. **implementation.py** (1206行) ⭐⭐⭐⭐⭐
**角色**: 工作流编排引擎（LangGraph 主文件）
**关键功能**:
```python
# 导出的关键组件
compiled_agent          # 编译后的 LangGraph 工作流
chatbot()              # 节点1：解析用户输入
visualization()        # 节点2：生成三角形可视化
analyze_with_llm()     # 节点3：LLM 热点分析
report()               # 节点4：生成最终报告
```

**依赖的模块**:
```
├── functions.py (所有核心算法)
├── config.py (配置管理)
├── excel_output_generator.py (CSV 输出)
├── llm_analyzer.py (LLM 调用)
├── data_preparers.py (数据准备)
├── visualization_comparison.py (可视化对比)
└── stub.py (LangGraph 框架支持)
```

**工作流节点**:
```
用户输入 → chatbot → visualization → analyze_with_llm → report → 输出
```

---

### 2. **functions.py** (2249行) ⭐⭐⭐⭐⭐
**角色**: 核心算法库（最大的文件）
**关键功能**:
- 交通数据处理
- 三角形/多边形生成
- DBSCAN 聚类
- 凸包计算
- 方向感知处理
- 道路信息加载

**主要函数**:
```python
process_direction_aware_traffic_data()  # 方向感知数据处理
cluster_triangles_dbscan()              # DBSCAN 聚类
calculate_all_hull_triangles()          # 计算凸包
visualize_triangles_with_hulls()        # 可视化
load_road_info()                        # 加载道路信息
```

---

### 3. **evaluation.py** (700行) ⭐⭐⭐⭐⭐
**角色**: 评估系统（刚修改过）
**关键功能**:
- 全局多边形 IoU
- 栅格化 F1-Score（基于 KP 区间） ← **新修改**
- 分层 IoU

**主要类**:
```python
class Evaluator:
    def __init__(road_type, direction, time_step_minutes)
    def evaluate_all_directions()
    def _evaluate_grid_f1()              # KP 区间栅格化
    def _rasterize_polygons_kp_based()   # 新增方法
```

---

### 4. **llm_analyzer.py** (328行) ⭐⭐⭐⭐
**角色**: LLM 调用封装
**关键功能**:
- OpenAI API 调用
- 批量热点分析
- Fallback 机制

**主要类**:
```python
class LLMAnalyzer:
    def analyze_traffic_data()
    def batch_analyze_directions()
```

---

### 5. **prompt_templates.py** (568行) ⭐⭐⭐⭐
**角色**: 提示词模板管理
**关键功能**:
- 构建分析提示词
- 少样本学习支持
- 自定义热点识别标准

**主要类**:
```python
class PromptTemplates:
    def build_analysis_prompt()
    def build_few_shot_examples()
```

---

### 6. **config.py** (122行) ⭐⭐⭐⭐
**角色**: 配置管理
**关键功能**:
- API 密钥管理
- 模型配置
- LLM 可用性检查

**主要类**:
```python
class Config:
    openai_api_key
    openai_model = "gpt-4o"
    is_llm_available()
```

---

### 7. **excel_output_generator.py** (545行) ⭐⭐⭐
**角色**: CSV/Excel 输出生成
**关键功能**:
- 生成预测 CSV
- 热点汇总表
- 道路信息集成

---

### 8. **data_preparers.py** (408行) ⭐⭐⭐
**角色**: 数据准备和格式化
**关键类**:
```python
class DataPreparer      # 数据预处理
class OutputFormatter   # 结果格式化
```

---

### 9. **visualization_comparison.py** (382行) ⭐⭐⭐
**角色**: 多方向对比可视化
**关键功能**:
- 上下行对比图
- 热点转换为凸包

---

### 10. **fallback_analyzer.py** (290行) ⭐⭐⭐
**角色**: 无 API 时的备用分析器
**关键功能**:
- 基于规则的热点识别
- 不需要 LLM

---

## 📊 依赖关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户入口脚本                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  test_complete_workflow.py        test_llm_workflow.py          │
│  batch_evaluation.py              test_evaluation.py            │
│  view_llm_thinking.py             create_few_shot_prompt.py     │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    核心工作流引擎                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              implementation.py (LangGraph 编排)                  │
│                                                                 │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐            │
│   │chatbot │→→→│visual. │→→→│LLM ana.│→→→│ report │            │
│   └────────┘   └────────┘   └────────┘   └────────┘            │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌───────────┐  ┌──────────────┐  ┌──────────────┐
│functions. │  │llm_analyzer. │  │evaluation.   │
│py         │  │py            │  │py            │
│           │  │              │  │              │
│(2249行)   │  │(328行)       │  │(700行)       │
│核心算法   │  │LLM调用       │  │评估系统      │
└───────────┘  └──────────────┘  └──────────────┘
     │              │                   │
     │              ▼                   │
     │      ┌──────────────┐            │
     │      │prompt_       │            │
     │      │templates.py  │            │
     │      │(568行)       │            │
     │      └──────────────┘            │
     │              │                   │
     ▼              ▼                   ▼
┌─────────────────────────────────────────────┐
│            支持模块                          │
├─────────────────────────────────────────────┤
│                                             │
│  config.py              (配置管理)          │
│  excel_output_generator.py (CSV输出)        │
│  data_preparers.py      (数据准备)          │
│  visualization_comparison.py (可视化)       │
│  fallback_analyzer.py   (备用分析)          │
│  stub.py                (框架支持)          │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🚀 推荐使用流程

### 场景1: 快速测试系统
```bash
python test_llm_workflow.py
```

### 场景2: 完整端到端测试
```bash
python test_complete_workflow.py
```

### 场景3: 测试评估系统
```bash
python test_evaluation.py --mode sample
```

### 场景4: 批量评估多个任务
```bash
# 1. 编辑 evaluation_config.json
# 2. 运行批量评估
python batch_evaluation.py --config evaluation_config.json
```

### 场景5: 查看 LLM 推理过程
```bash
python view_llm_thinking.py output/workflow_state.json
```

---

## 🔍 文件大小排名（行数）

1. **functions.py** - 2249 行（核心算法）
2. **implementation.py** - 1206 行（工作流）
3. **evaluation.py** - 700 行（评估）
4. **prompt_templates.py** - 568 行（提示词）
5. **excel_output_generator.py** - 545 行（输出）
6. **batch_evaluation.py** - 439 行（批量评估）
7. **test_complete_workflow.py** - 432 行（测试）
8. **data_preparers.py** - 408 行（数据准备）
9. **visualization_comparison.py** - 382 行（可视化）
10. **extract_few_shot_examples.py** - 332 行（少样本提取）

---

## 📝 测试脚本分类

### 功能测试
- `test_complete_workflow.py` - 完整流程
- `test_llm_workflow.py` - LLM 分析
- `test_evaluation.py` - 评估系统

### 调试/修复测试
- `test_clustering_fix.py` - 聚类修复
- `test_direction_fix.py` - 方向修复
- `test_enhanced_prompt.py` - 提示词增强
- `test_new_format.py` - 新格式测试
- `test_shape_normalization.py` - 形状归一化
- `test_token_fix.py` - Token 修复

### 工具脚本
- `view_llm_thinking.py` - 查看思考链
- `create_few_shot_prompt.py` - 创建少样本
- `extract_few_shot_examples.py` - 提取示例

---

## 🎯 核心数据流

```
用户输入（自然语言）
    ↓
chatbot 节点 → 解析出: route, direction, date
    ↓
加载 CSV 文件（历史拥堵数据）
    ↓
visualization 节点 → 生成三角形 + 聚类 + 凸包
    ↓
analyze_with_llm 节点 → LLM 识别热点
    ↓
report 节点 → 生成最终报告
    ↓
输出：
  - CSV 预测文件
  - 可视化图表
  - Markdown 报告
  - workflow_state.json
```

---

## 🔧 配置文件

- `.env` - 环境变量（API 密钥）
- `evaluation_config.json` - 批量评估配置
- `roadic_kannetsu.csv` - 道路信息（関越道）
- `roadic_touhoku.csv` - 道路信息（東北道）

---

## 📂 输出目录

```
output/
├── triangles_*.png           # 三角形可视化
├── comparison_*.png          # 多方向对比图
├── congestion_prediction_*.csv  # 预测输出
├── workflow_state.json       # 工作流状态快照
└── evaluation/               # 评估结果
    ├── *_evaluation.json
    └── batch_summary.json
```

---

*最后更新：2025-11-10*
