# 评估系统快速开始

## 🚀 5分钟快速开始

### 步骤1: 安装依赖
```bash
pip install shapely
```

### 步骤2: 测试评估功能
```bash
# 使用模拟数据测试
python test_evaluation.py --mode sample
```

如果看到类似以下输出，说明安装成功：
```
【指标1】全局多边形IoU: 0.4796
【指标2】栅格化评估 F1-Score: 0.7500
【指标3】分层IoU 综合: 0.4694
✓ 测试完成！
```

### 步骤3: 配置评估任务

编辑 `evaluation_config.json`，添加你的评估任务：

```json
{
  "tasks": [
    {
      "task_id": "your_task_name",
      "route": "関越道",
      "direction": "下",
      "target_date": "2025-05-05",
      "ground_truth_file": "data/processed_data/関越道_下_2025_05-05.csv"
    }
  ]
}
```

### 步骤4: 运行批量评估

```bash
python batch_evaluation.py --config evaluation_config.json
```

### 步骤5: 查看结果

结果保存在 `output/evaluation/` 目录：
- `{task_id}_evaluation.json`: 单个任务详细结果
- `batch_summary.json`: 所有任务汇总

## 📊 评估指标解读

### 指标1: 全局多边形IoU
- **范围**: 0-1
- **含义**: 预测区域与真实区域的重叠程度
- **好坏**:
  - > 0.7: 优秀
  - 0.5-0.7: 良好
  - 0.3-0.5: 一般
  - < 0.3: 较差

### 指标2: 栅格F1-Score
- **范围**: 0-1
- **含义**: 综合考虑Precision和Recall的平衡指标
- **Precision高, Recall低**: 预测保守，漏报多
- **Precision低, Recall高**: 预测激进，误报多
- **好坏**:
  - > 0.8: 优秀
  - 0.6-0.8: 良好
  - 0.4-0.6: 一般
  - < 0.4: 较差

### 指标3: 分层IoU
- **空间IoU**: KP维度的重叠（位置准确性）
- **时间IoU**: 时间维度的重叠（时间准确性）
- **综合IoU**: 两者的乘积

**诊断用途**:
- 空间高, 时间低 → 位置准确但时间预测有偏差
- 空间低, 时间高 → 时间准确但位置预测有偏差
- 两者都低 → 整体预测质量需要改进

## 🔧 常用命令

### 测试评估功能
```bash
# 模拟数据测试
python test_evaluation.py --mode sample

# 真实数据测试
python test_evaluation.py --mode real
```

### 批量评估
```bash
# 使用默认配置
python batch_evaluation.py

# 使用自定义配置
python batch_evaluation.py --config my_config.json
```

### 使用快捷脚本（Linux/Mac）
```bash
./run_evaluation.sh
```

## 📁 数据文件准备

### 真实数据格式要求

CSV文件需要包含以下列：
- `date`: 日期（如2025-05-05）
- `上下`: 方向（上/下）
- `原因`: 原因（如"交通集中"）
- `道路番号`: 路线名（如"関越道"）
- `発生時刻`: 发生时刻
- `ピーク時刻`: 峰值时刻
- `ピーク長`: 峰值长度
- `発生Ｋｐ`: 发生KP
- `発生時渋滞長`: 发生时拥堵长度
- `渋滞時間`: 拥堵时间

示例数据位置：
```
data/processed_data/関越道_下_2025_05-05.csv
```

## 🐛 问题排查

### 问题1: ModuleNotFoundError: shapely
**解决**: 安装shapely
```bash
pip install shapely
```

### 问题2: 真实数据文件不存在
**解决**: 检查配置文件中的路径是否正确
```json
"ground_truth_file": "data/processed_data/関越道_下_2025_05-05.csv"
```

### 问题3: 预测workflow未返回结果
**原因**: 可能是LLM调用失败或数据处理错误
**解决**:
1. 检查 `.env` 文件中的API配置
2. 查看控制台输出的错误信息
3. 确认训练数据文件存在

### 问题4: 所有指标都是0
**原因**: 预测和真实数据完全不重叠
**解决**:
1. 检查预测结果是否合理
2. 检查真实数据是否正确加载
3. 确认方向（上/下）是否匹配

## 💡 最佳实践

### 1. 先用模拟数据测试
在使用真实数据前，先运行 `test_evaluation.py --mode sample` 确保系统正常。

### 2. 逐个任务测试
不要一次添加太多任务，先测试单个任务确保配置正确。

### 3. 检查输出目录
定期清理 `output/evaluation/` 目录，避免结果混淆。

### 4. 保存配置文件
为不同实验保存不同的配置文件：
- `evaluation_config_exp1.json`
- `evaluation_config_exp2.json`

### 5. 对比多个模型
通过批量评估对比不同模型版本的性能。

## 📖 进阶使用

### 在Python代码中使用

```python
from evaluation import Evaluator

# 创建评估器
evaluator = Evaluator(grid_resolution=(1.0, 60))

# 准备数据
predictions = {"下": [...]}
ground_truth = {"下": [...]}

# 评估
results = evaluator.evaluate_all_directions(predictions, ground_truth)

# 打印报告
evaluator.print_summary_report(results)

# 保存结果
evaluator.save_results_to_json(results, "my_result.json")
```

### 自定义栅格分辨率

```python
# 更精细
evaluator = Evaluator(grid_resolution=(0.5, 30))  # 0.5km × 30min

# 更粗糙
evaluator = Evaluator(grid_resolution=(2.0, 120))  # 2km × 2h
```

## 📧 技术支持

详细文档请参考: `EVALUATION_GUIDE.md`

---

*快速开始指南 - 2025年10月*
