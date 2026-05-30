# Version 2 - 精简版工作流测试

## 📁 目录说明

这是从 `version1_3` 精简而来的版本，只保留了运行 `test_llm_workflow.py` 所需的核心文件，并增强了评估功能。

## 📄 包含的文件

### 核心文件（12个）

1. **test_llm_workflow.py** - 主测试脚本（已增强，包含评估功能）
2. **implementation.py** - LangGraph 工作流引擎
3. **functions.py** - 核心算法库（2249行）
4. **config.py** - 配置管理
5. **evaluation.py** - 评估系统（新的KP区间栅格化）
6. **llm_analyzer.py** - LLM 调用封装
7. **prompt_templates.py** - 提示词模板
8. **data_preparers.py** - 数据准备
9. **excel_output_generator.py** - CSV 输出
10. **visualization_comparison.py** - 可视化对比
11. **fallback_analyzer.py** - 备用分析器
12. **stub.py** - LangGraph 框架支持

### 配置文件

- **.env** - 环境变量（API密钥）

## 🚀 使用方法

### 1. 基本运行（不评估）

```bash
cd /home/dizhihuang/graduate/predict_workflow/version2
python test_llm_workflow.py
```

这将运行完整的预测工作流，但不进行评估。

### 2. 带评估运行

```bash
python test_llm_workflow.py --gt /path/to/ground_truth.csv
```

示例：
```bash
# 使用2025年5月3日的真实数据进行评估
python test_llm_workflow.py --gt ../data/processed_data/関越道_下_2025_05-03.csv
```

## 📊 新增功能

### 评估系统集成

`test_llm_workflow.py` 现在包含：

1. **预测阶段**：运行LLM分析工作流
   - 解析用户输入
   - 生成三角形可视化
   - LLM热点识别
   - 生成报告

2. **评估阶段**（可选）：
   - 加载 ground truth 数据
   - 计算三种 IoU 指标：
     - 全局多边形 IoU
     - 栅格化 F1-Score（基于KP区间）
     - 分层 IoU（空间+时间）
   - 生成评估报告
   - 保存评估结果到 JSON

## 📈 输出说明

### 预测输出

```
output/
├── triangles_関越道_下_20241110_*.png     # 三角形可视化
├── comparison_関越道_*.png                # 多方向对比图
├── congestion_prediction_*.csv           # 预测CSV
└── workflow_state.json                   # 工作流状态
```

### 评估输出（如果提供了ground truth）

```
output/evaluation/
└── evaluation_関越道_下_20241110_*.json  # 评估结果
```

## 🔧 参数说明

```bash
python test_llm_workflow.py [OPTIONS]

Options:
  --gt PATH    Ground truth CSV文件路径（可选）
               如果不提供，则只运行预测，不评估
```

## 📝 示例输出

### 不带评估

```
🧪 Testing LLM-based Workflow with Evaluation
======================================================================

📋 Configuration:
   API Key: Set ✅
   Model: gpt-4o
   LLM Available: True

📝 Test Input:
   请分析関越高速公路下行方向2024年5月3日的交通情况

======================================================================
🚀 Running workflow...
======================================================================

[... workflow output ...]

📊 Prediction Results:

🤖 LLM Analysis:
   下 direction:
      Hotspots: 3
      Confidence: 0.85
      Hotspot 1: KP 29.0-35.0km, frequency: 5
      ...

💡 未提供Ground Truth文件，跳过评估
   提示：使用 --gt 参数指定ground truth文件
```

### 带评估

```
[... prediction output ...]

📂 加载Ground Truth数据: ../data/processed_data/関越道_下_2025_05-03.csv
✓ Ground Truth加载完成: 15 个拥堵事件

======================================================================
📊 开始评估预测结果
======================================================================

✓ 已加载 68 个KP区间

==================================================
评估方向: 下
==================================================

【指标1】全局多边形IoU: 0.4523
  - 预测区域总面积: 520.00 km·min
  - 真实区域总面积: 345.21 km·min
  - 交集面积: 280.15 km·min
  - 并集面积: 585.06 km·min

【指标2】栅格化评估 (KP区间: 68个 × 时间: 60min)
  - Precision: 0.6500 (预测的65.0%是正确的)
  - Recall: 0.7800 (真实拥堵的78.0%被预测到)
  - F1-Score: 0.7090
  - Grid IoU: 0.5487

【指标3】分层IoU
  - 空间IoU (KP): 0.6234
  - 时间IoU (Time): 0.7012
  - 综合IoU (相乘): 0.4371

======================================================================
📈 Evaluation Summary
======================================================================

【下行】
  多边形IoU:      0.4523
  F1-Score:       0.7090
  Precision:      0.6500
  Recall:         0.7800
```

## 🎯 主要改进

相比原始 `test_llm_workflow.py`：

1. **集成评估功能**：无需单独运行评估脚本
2. **命令行参数**：通过 `--gt` 参数灵活控制是否评估
3. **更详细的输出**：显示评估摘要
4. **自动保存结果**：评估结果自动保存为JSON

## ⚙️ 工作流程

```
用户输入
    ↓
LangGraph工作流 (implementation.py)
    ├─ chatbot节点 → 解析输入
    ├─ visualization节点 → 生成三角形
    ├─ analyze_with_llm节点 → LLM分析
    └─ report节点 → 生成报告
    ↓
[可选] 评估模块 (evaluation.py)
    ├─ 加载ground truth
    ├─ 计算IoU指标
    └─ 保存评估结果
    ↓
输出完成
```

## 📌 注意事项

1. **API密钥**：确保 `.env` 文件中配置了 `OPENAI_API_KEY`
2. **数据路径**：ground truth 文件必须是标准的交通拥堵CSV格式
3. **方向匹配**：确保测试输入中的方向（上/下）与ground truth数据的方向一致
4. **依赖库**：需要安装所有必要的Python库（numpy, pandas, shapely, matplotlib等）

## 🔗 相关文件

- 数据目录：`/home/dizhihuang/graduate/predict_workflow/data/`
- 道路信息：`../data/roadic_kannetsu.csv`（関越道）
- 原版本：`../version1_3/`

---

*创建于 2025-11-10*
