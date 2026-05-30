# 评估系统使用指南

## 概述

本评估系统用于评估交通拥堵预测模型的准确性，通过对比预测结果与2025年真实数据，计算三种IoU指标。

## 文件说明

```
version1_3/
├── evaluation.py              # 核心评估模块（三种IoU指标）
├── batch_evaluation.py        # 批量评估脚本
├── evaluation_config.json     # 批量评估配置文件
├── test_evaluation.py         # 评估功能测试脚本
└── EVALUATION_GUIDE.md        # 本文档
```

## 三种评估指标

### 指标1：全局多边形IoU
- **定义**: IoU = Area(预测 ∩ 真实) / Area(预测 ∪ 真实)
- **含义**: 衡量预测区域与真实拥堵区域的整体重叠程度
- **值域**: 0-1，越高越好
- **优点**: 直观，整体评估
- **适用**: 评估预测是否覆盖了真实拥堵区域

### 指标2：栅格化F1-Score
- **定义**:
  - 将空间划分为1km × 1h的栅格
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **含义**:
  - Precision: 预测的栅格中有多少是正确的（避免误报）
  - Recall: 真实拥堵的栅格中有多少被预测到（避免漏报）
  - F1: 综合平衡指标
- **值域**: 0-1，越高越好
- **优点**: 可以分别看出"误报"和"漏报"问题
- **适用**: 诊断模型倾向（过度预测 vs 保守预测）

### 指标3：分层IoU（空间 + 时间）
- **定义**:
  - 空间IoU: 在KP维度（横轴）计算1维区间IoU
  - 时间IoU: 在时间维度（纵轴）计算1维区间IoU
  - 综合IoU: 空间IoU × 时间IoU
- **含义**: 分别评估空间预测和时间预测的准确性
- **值域**: 0-1，越高越好
- **优点**: 可以诊断是"位置预测错误"还是"时间预测错误"
- **适用**: 模型调优，定位问题所在

## 使用方法

### 方法1：测试评估模块

测试评估功能是否正常工作：

```bash
# 使用模拟数据测试
python test_evaluation.py --mode sample

# 使用真实数据测试（需要真实数据文件）
python test_evaluation.py --mode real
```

### 方法2：批量评估

#### 步骤1：准备配置文件

编辑 `evaluation_config.json`：

```json
{
  "tasks": [
    {
      "task_id": "sekietsu_2025_05_05_down",
      "route": "関越道",
      "direction": "下",
      "target_date": "2025-05-05",
      "train_years": [2014, 2018, 2024],
      "ground_truth_file": "data/processed_data/関越道_下_2025_05-05.csv"
    },
    {
      "task_id": "sekietsu_2025_05_05_up",
      "route": "関越道",
      "direction": "上",
      "target_date": "2025-05-05",
      "train_years": [2014, 2018, 2024],
      "ground_truth_file": "data/processed_data/関越道_上_2025_05-05.csv"
    }
  ]
}
```

**配置字段说明**：
- `task_id`: 任务唯一标识符（用于输出文件命名）
- `route`: 路线名称（如"関越道"）
- `direction`: 方向（"上" 或 "下"）
- `target_date`: 预测目标日期（格式：YYYY-MM-DD）
- `train_years`: 训练数据年份列表
- `ground_truth_file`: 2025年真实数据文件路径

#### 步骤2：运行批量评估

```bash
python batch_evaluation.py --config evaluation_config.json
```

#### 步骤3：查看结果

评估结果保存在 `output/evaluation/` 目录：

```
output/evaluation/
├── sekietsu_2025_05_05_down_evaluation.json  # 单个任务详细结果
├── sekietsu_2025_05_05_up_evaluation.json    # 单个任务详细结果
└── batch_summary.json                        # 所有任务汇总结果
```

### 方法3：在代码中使用评估器

```python
from evaluation import Evaluator

# 创建评估器（1km × 1h栅格）
evaluator = Evaluator(grid_resolution=(1.0, 60))

# 准备数据
predictions = {
    "下": [
        {
            "hotspot_id": 1,
            "prediction_shape": {
                "vertices": [[29.0, 480], [29.0, 600], [35.0, 540]]
            }
        }
    ]
}

ground_truth = {
    "下": [
        {
            "vertices": [[30.0, 490], [30.0, 580], [34.0, 535]],
            "kp_start": 30.0,
            "kp_end": 34.0,
            "time_start": 490,
            "time_end": 580
        }
    ]
}

# 执行评估
results = evaluator.evaluate_all_directions(predictions, ground_truth)

# 打印报告
evaluator.print_summary_report(results)

# 保存结果
evaluator.save_results_to_json(results, "output/my_evaluation.json")
```

## 输出格式

### 控制台输出示例

```
==================================================
评估方向: 下
==================================================

【指标1】全局多边形IoU: 0.4796
  - 预测区域总面积: 480.00 km·min
  - 真实区域总面积: 230.21 km·min
  - 交集面积: 230.21 km·min
  - 并集面积: 480.00 km·min

【指标2】栅格化评估 (分辨率: 1.0km × 60min)
  - Precision: 0.6000 (预测的60.0%是正确的)
  - Recall: 1.0000 (真实拥堵的100.0%被预测到)
  - F1-Score: 0.7500
  - Grid IoU: 0.6000

【指标3】分层IoU
  - 空间IoU (KP): 0.6500
  - 时间IoU (Time): 0.7222
  - 综合IoU (相乘): 0.4694
```

### JSON输出格式

```json
{
  "evaluation_time": "2025-10-08 12:34:56",
  "grid_resolution": {
    "kp_step_km": 1.0,
    "time_step_min": 60
  },
  "results": {
    "下": {
      "polygon_iou": 0.4796,
      "grid_metrics": {
        "precision": 0.6000,
        "recall": 1.0000,
        "f1_score": 0.7500,
        "grid_iou": 0.6000
      },
      "spatial_iou": 0.6500,
      "temporal_iou": 0.7222,
      "combined_iou": 0.4694,
      "pred_count": 2,
      "gt_count": 3
    },
    "average": { ... }
  }
}
```

## 常见问题

### Q1: 为什么三个指标的值不一样？

这是正常的，因为它们衡量的角度不同：
- **多边形IoU**: 几何面积重叠，受形状影响较大
- **栅格F1**: 栅格级别匹配，受分辨率影响
- **分层IoU**: 1维投影重叠，丢失了时空耦合信息

建议综合考虑三个指标，而非只看单一指标。

### Q2: 如何选择栅格分辨率？

默认 1km × 1h 是基于以下考虑：
- **1km**: 与高速公路出入口间距相当
- **1h**: 拥堵持续时间的合理粒度

可以根据需要调整：
```python
# 更精细（计算更慢）
evaluator = Evaluator(grid_resolution=(0.5, 30))  # 0.5km × 30min

# 更粗糙（计算更快）
evaluator = Evaluator(grid_resolution=(2.0, 120))  # 2km × 2h
```

### Q3: 如果预测和真实数据完全不重叠怎么办？

所有指标都会是0.0，这说明预测完全失败。可以：
1. 检查数据格式是否正确
2. 检查预测的KP范围和时间范围是否合理
3. 分析为什么预测偏差如此大

### Q4: 批量评估失败怎么办？

检查：
1. 配置文件格式是否正确（JSON语法）
2. `ground_truth_file` 路径是否存在
3. CSV文件格式是否与训练数据一致
4. 查看 `batch_summary.json` 中的错误信息

### Q5: 如何添加更多评估任务？

在 `evaluation_config.json` 的 `tasks` 数组中添加：

```json
{
  "tasks": [
    { "task_id": "task1", ... },
    { "task_id": "task2", ... },
    { "task_id": "task3", ... }  // 新增任务
  ]
}
```

每个任务会独立运行预测和评估。

## 依赖项

需要安装以下Python包：

```bash
pip install shapely numpy pandas matplotlib
```

如果已经能运行预测系统，只需额外安装：

```bash
pip install shapely
```

## 扩展功能

### 可视化对比（计划中）

未来可以添加可视化功能，生成预测vs真实的对比图：
- 栅格热图（TP/FP/FN着色）
- 多边形叠加图
- 误差分布图

### 多分辨率评估（可选）

可以尝试多个栅格分辨率，评估结果稳定性。

## 技术支持

如有问题，请检查：
1. `test_evaluation.py` 是否能通过测试
2. 日志输出中的警告信息
3. 数据格式是否正确

---

*评估系统开发完成于2025年10月*
