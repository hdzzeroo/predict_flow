# 评估指标详解

## 概述

本文档详细说明交通拥堵预测评估系统中使用的三种IoU评估指标的原理、计算方式和应用场景。

---

## 指标1：全局多边形IoU (Global Polygon IoU)

### 原理

**核心思想**：将所有预测区域和所有真实拥堵区域看作二维平面上的多边形，计算它们的几何重叠程度。

IoU (Intersection over Union) 是目标检测领域的经典指标，定义为：

```
IoU = Area(A ∩ B) / Area(A ∪ B)
```

其中：
- A: 预测区域的并集
- B: 真实区域的并集
- ∩: 交集操作
- ∪: 并集操作

### 计算步骤

#### 步骤1: 合并所有预测多边形
```
输入：N个预测热点多边形 P₁, P₂, ..., Pₙ
操作：Pred_Union = P₁ ∪ P₂ ∪ ... ∪ Pₙ
```

**示例**：
- 预测热点1：三角形，顶点 [[29, 480], [29, 600], [35, 540]]
- 预测热点2：三角形，顶点 [[4, 420], [4, 480], [8, 450]]
- Pred_Union：这两个三角形的并集（可能是两个不相连的区域）

#### 步骤2: 合并所有真实多边形
```
输入：M个真实拥堵三角形 T₁, T₂, ..., Tₘ
操作：GT_Union = T₁ ∪ T₂ ∪ ... ∪ Tₘ
```

#### 步骤3: 计算交集面积
```
Intersection = Pred_Union ∩ GT_Union
Intersection_Area = Area(Intersection)
```

**几何意义**：预测区域和真实区域重叠的部分

#### 步骤4: 计算并集面积
```
Union = Pred_Union ∪ GT_Union
Union_Area = Area(Union)
```

**几何意义**：预测区域和真实区域覆盖的总范围

#### 步骤5: 计算IoU
```
Polygon_IoU = Intersection_Area / Union_Area
```

### 数学表达

在二维坐标系(KP, Time)中：

```
Polygon_IoU = ∫∫[1_Pred(x,y) ∧ 1_GT(x,y)] dx dy / ∫∫[1_Pred(x,y) ∨ 1_GT(x,y)] dx dy
```

其中：
- 1_Pred(x,y): 点(x,y)是否在预测区域内的指示函数
- 1_GT(x,y): 点(x,y)是否在真实区域内的指示函数
- ∧: 逻辑AND（交集）
- ∨: 逻辑OR（并集）

### 计算示例

**场景**：预测1个大三角形 vs 真实3个小三角形

```
预测多边形：
  P = Triangle(vertices=[(29, 480), (29, 600), (35, 540)])
  Area(P) = 0.5 × base × height = 0.5 × 120 × 6 = 360 km·min

真实多边形：
  T₁ = Triangle([(30, 490), (30, 580), (34, 535)]), Area = 180 km·min
  T₂ = Triangle([(31, 500), (31, 560), (33, 530)]), Area = 60 km·min
  T₃ = Triangle([(4, 430), (4, 470), (7, 450)]), Area = 60 km·min

  GT_Union = T₁ ∪ T₂ ∪ T₃
  Area(GT_Union) ≈ 280 km·min (有重叠，所以小于180+60+60)

交集：
  Intersection = P ∩ GT_Union
  由于T₃不在P的范围内，只有T₁和T₂与P重叠
  Area(Intersection) ≈ 220 km·min

并集：
  Union = P ∪ GT_Union
  Area(Union) = Area(P) + Area(GT_Union) - Area(Intersection)
              = 360 + 280 - 220 = 420 km·min

IoU：
  Polygon_IoU = 220 / 420 ≈ 0.524
```

### 值域与解释

| IoU值 | 解释 | 质量评价 |
|-------|------|---------|
| 1.0 | 完全重合 | 完美 |
| 0.7-0.9 | 高度重叠 | 优秀 |
| 0.5-0.7 | 中度重叠 | 良好 |
| 0.3-0.5 | 轻度重叠 | 一般 |
| 0.1-0.3 | 微弱重叠 | 较差 |
| 0.0 | 完全不重叠 | 失败 |

### 优点
- ✅ 直观易懂，符合人类对"重叠"的理解
- ✅ 不受区域数量影响（预测1个或10个热点都公平）
- ✅ 对位置和形状都敏感
- ✅ 业界标准指标，便于对比

### 缺点
- ⚠️ 无法区分"预测过度"和"预测不足"
- ⚠️ 对小偏移敏感（预测位置略偏，IoU可能大幅下降）
- ⚠️ 计算复杂多边形的交并集需要几何库支持

### 实现技术

使用Python的`shapely`库：
```python
from shapely.geometry import Polygon
from shapely.ops import unary_union

# 合并多边形
pred_union = unary_union(pred_polygons)
gt_union = unary_union(gt_polygons)

# 计算交集和并集
intersection = pred_union.intersection(gt_union)
union = pred_union.union(gt_union)

# 计算IoU
iou = intersection.area / union.area
```

---

## 指标2：栅格化F1-Score (Grid-based F1-Score)

### 原理

**核心思想**：将连续的时空平面离散化为栅格（网格），将预测和评估转换为二分类问题，每个栅格单元要么"有拥堵"(1)要么"无拥堵"(0)。

F1-Score是机器学习中的经典分类指标，综合考虑精确率(Precision)和召回率(Recall)。

### 计算步骤

#### 步骤1: 确定坐标范围
```
kp_min = min(所有多边形的最小KP)
kp_max = max(所有多边形的最大KP)
time_min = min(所有多边形的最小时间)
time_max = max(所有多边形的最大时间)
```

**示例**：
- KP范围: [4.0, 35.0] km
- 时间范围: [420, 600] 分钟 (7:00 - 10:00)

#### 步骤2: 创建栅格网格
```
栅格分辨率：kp_step = 1.0 km, time_step = 60 min

栅格数量：
  n_kp = ⌈(kp_max - kp_min) / kp_step⌉ = ⌈31 / 1.0⌉ = 31
  n_time = ⌈(time_max - time_min) / time_step⌉ = ⌈180 / 60⌉ = 3

总栅格单元数：31 × 3 = 93 个
```

栅格示例：
```
时间 ^
600  |  [ ][ ][ ]...[X]  ← 第3行
540  |  [ ][X][X]...[X]  ← 第2行
480  |  [ ][X][X]...[X]  ← 第1行
420  |__________________> KP
     4  5  6  7 ... 35

[X] = 有拥堵
[ ] = 无拥堵
```

#### 步骤3: 栅格化预测多边形
```
pred_grid = zeros((n_kp, n_time))

对于每个栅格单元(i, j)：
    中心点坐标：
      kp_center = kp_min + (i + 0.5) × kp_step
      time_center = time_min + (j + 0.5) × time_step

    如果 Point(kp_center, time_center) 在任何预测多边形内：
      pred_grid[i, j] = 1
    否则：
      pred_grid[i, j] = 0
```

#### 步骤4: 栅格化真实多边形
```
gt_grid = zeros((n_kp, n_time))

（与步骤3相同，但使用真实多边形）
```

#### 步骤5: 计算混淆矩阵
```
TP (True Positive)  = count(pred_grid == 1 AND gt_grid == 1)  # 正确预测有拥堵
FP (False Positive) = count(pred_grid == 1 AND gt_grid == 0)  # 误报（实际无拥堵）
FN (False Negative) = count(pred_grid == 0 AND gt_grid == 1)  # 漏报（实际有拥堵）
TN (True Negative)  = count(pred_grid == 0 AND gt_grid == 0)  # 正确预测无拥堵
```

**混淆矩阵可视化**：
```
                实际情况
              有拥堵  无拥堵
预测   有拥堵    TP      FP
      无拥堵    FN      TN
```

#### 步骤6: 计算评估指标
```
Precision (精确率) = TP / (TP + FP)
  → 预测为"有拥堵"的栅格中，真正有拥堵的比例
  → 反映"预测的准确性"

Recall (召回率) = TP / (TP + FN)
  → 实际有拥堵的栅格中，被预测出来的比例
  → 反映"预测的完整性"

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
  → 精确率和召回率的调和平均
  → 综合平衡指标

Grid_IoU = TP / (TP + FP + FN)
  → 栅格级别的IoU（与多边形IoU对应）
```

### 数学表达

```
Precision = |Pred ∩ GT| / |Pred|
Recall = |Pred ∩ GT| / |GT|
F1 = 2 × Precision × Recall / (Precision + Recall)
```

其中|·|表示集合的基数（栅格数量）。

### 计算示例

**场景**：预测覆盖60个栅格，真实覆盖50个栅格

```
栅格统计：
  总栅格数：93个
  预测为1的栅格：60个
  真实为1的栅格：50个
  重叠的栅格：36个

混淆矩阵：
  TP = 36  (预测对了)
  FP = 24  (误报：预测有但实际无)
  FN = 14  (漏报：预测无但实际有)
  TN = 19  (正确预测无拥堵)

计算指标：
  Precision = 36 / (36 + 24) = 36 / 60 = 0.60
    → 预测的60个栅格中，只有60%是正确的
    → 说明有40%的误报（过度预测）

  Recall = 36 / (36 + 14) = 36 / 50 = 0.72
    → 真实的50个拥堵栅格中，72%被预测到
    → 说明有28%的漏报（预测不足）

  F1-Score = 2 × 0.60 × 0.72 / (0.60 + 0.72) = 0.655
    → 综合评分65.5%

  Grid_IoU = 36 / (36 + 24 + 14) = 36 / 74 = 0.486
```

### 指标解读

#### Precision vs Recall 的含义

| Precision | Recall | 解释 | 问题 |
|-----------|--------|------|------|
| 高 | 高 | 预测准确且完整 | 理想状态 ✅ |
| 高 | 低 | 预测准确但不完整 | 漏报多，预测保守 ⚠️ |
| 低 | 高 | 预测完整但不准确 | 误报多，预测激进 ⚠️ |
| 低 | 低 | 预测既不准也不全 | 模型失败 ❌ |

#### F1-Score的价值

F1-Score是调和平均，对Precision和Recall的权重相等：
```
当Precision = Recall时，F1 = Precision = Recall
当Precision ≠ Recall时，F1更接近较小的那个
```

**为什么用调和平均而非算术平均？**
- 算术平均：(0.9 + 0.1) / 2 = 0.5
- 调和平均：2 × 0.9 × 0.1 / (0.9 + 0.1) = 0.18

调和平均对极端不平衡更敏感，避免"一高一低"被平均掉。

### 值域与解释

| F1-Score | 质量评价 |
|----------|---------|
| 0.9-1.0 | 优秀 |
| 0.7-0.9 | 良好 |
| 0.5-0.7 | 一般 |
| 0.3-0.5 | 较差 |
| 0.0-0.3 | 失败 |

### 优点
- ✅ 可以分别看Precision和Recall，诊断问题类型
- ✅ F1综合平衡两者，单一指标评估
- ✅ 计算简单高效
- ✅ 对栅格分辨率可调，灵活性高

### 缺点
- ⚠️ 结果受栅格分辨率影响（太粗或太细都不好）
- ⚠️ 对小尺度偏移敏感（位置偏移可能导致栅格完全不重叠）
- ⚠️ 忽略了预测的置信度（只看是否预测，不看预测强度）

### 分辨率选择建议

| 场景 | KP步长 | 时间步长 | 说明 |
|------|--------|---------|------|
| 宏观评估 | 2-5 km | 1-2 h | 粗粒度，计算快 |
| **标准评估** | **1 km** | **1 h** | **推荐** ✅ |
| 精细评估 | 0.5 km | 30 min | 细粒度，计算慢 |

### 实现技术

```python
import numpy as np
from shapely.geometry import Point

def rasterize(polygons, bounds, kp_step, time_step):
    kp_min, kp_max, time_min, time_max = bounds
    n_kp = int(np.ceil((kp_max - kp_min) / kp_step))
    n_time = int(np.ceil((time_max - time_min) / time_step))

    grid = np.zeros((n_kp, n_time), dtype=np.uint8)
    union_poly = unary_union(polygons)

    for i in range(n_kp):
        for j in range(n_time):
            kp = kp_min + (i + 0.5) * kp_step
            time = time_min + (j + 0.5) * time_step
            point = Point(kp, time)

            if union_poly.contains(point):
                grid[i, j] = 1

    return grid

# 计算混淆矩阵
tp = np.sum((pred_grid == 1) & (gt_grid == 1))
fp = np.sum((pred_grid == 1) & (gt_grid == 0))
fn = np.sum((pred_grid == 0) & (gt_grid == 1))

# 计算指标
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
```

---

## 指标3：分层IoU (Layered IoU)

### 原理

**核心思想**：将二维时空问题分解为两个独立的一维问题，分别在空间维度(KP)和时间维度(Time)上计算IoU。

这种方法牺牲了时空耦合信息，换取了计算简单和可解释性。

### 为什么需要分层IoU？

**问题场景**：
- 预测"早上8点在KP30"有拥堵
- 实际"下午3点在KP30"有拥堵

多边形IoU会很低（时间完全不重叠），但我们可以看到：
- 空间预测是对的（都在KP30）
- 时间预测是错的（相差7小时）

分层IoU可以帮助**定位问题所在**。

### 计算步骤

#### 步骤1: 提取空间区间（KP维度）

**对于预测多边形**：
```
对每个预测多边形P：
  kp_min_P = min(P的所有顶点的KP坐标)
  kp_max_P = max(P的所有顶点的KP坐标)
  区间_P = [kp_min_P, kp_max_P]

所有预测区间：
  Pred_KP_intervals = [区间_1, 区间_2, ...]
```

**示例**：
```
预测热点1：Triangle([(29, 480), (29, 600), (35, 540)])
  → KP区间：[29, 35]

预测热点2：Triangle([(4, 420), (4, 480), (8, 450)])
  → KP区间：[4, 8]

Pred_KP_intervals = [[29, 35], [4, 8]]
```

**对于真实多边形**（同理）：
```
GT_KP_intervals = [[30, 34], [31, 33], [4.5, 7]]
```

#### 步骤2: 合并重叠区间（空间）

```
对Pred_KP_intervals和GT_KP_intervals分别排序并合并重叠区间：

Pred_KP_merged = merge([[4, 8], [29, 35]])
               = [[4, 8], [29, 35]]  (无重叠，保持不变)

GT_KP_merged = merge([[4.5, 7], [30, 34], [31, 33]])
             = [[4.5, 7], [30, 34]]  (后两个有重叠，合并)
```

**合并算法**：
```python
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # 有重叠
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged
```

#### 步骤3: 计算空间IoU

```
空间交集长度 = intervals_intersection(Pred_KP_merged, GT_KP_merged)
空间并集长度 = intervals_union(Pred_KP_merged, GT_KP_merged)

Spatial_IoU = 空间交集长度 / 空间并集长度
```

**详细计算**：
```
Pred_KP: [[4, 8], [29, 35]]  → 总长度: 4 + 6 = 10 km
GT_KP:   [[4.5, 7], [30, 34]] → 总长度: 2.5 + 4 = 6.5 km

交集计算：
  [4, 8] ∩ [4.5, 7] = [4.5, 7]  → 长度2.5 km
  [29, 35] ∩ [30, 34] = [30, 34] → 长度4 km
  总交集长度 = 2.5 + 4 = 6.5 km

并集计算：
  合并所有区间：[[4, 8], [29, 35]] ∪ [[4.5, 7], [30, 34]]
  = [[4, 8], [29, 35]]  (第二组完全被第一组包含)
  总并集长度 = 4 + 6 = 10 km

Spatial_IoU = 6.5 / 10 = 0.65
```

#### 步骤4: 提取时间区间（Time维度）

（与空间维度计算完全类似）

```
预测时间区间：
  热点1：[480, 600]  (8:00-10:00)
  热点2：[420, 480]  (7:00-8:00)
  Pred_Time_merged = [[420, 600]]  (合并后)

真实时间区间：
  拥堵1：[490, 580]
  拥堵2：[500, 560]
  拥堵3：[430, 470]
  GT_Time_merged = [[430, 580]]  (合并后)
```

#### 步骤5: 计算时间IoU

```
时间交集：[420, 600] ∩ [430, 580] = [430, 580] → 长度150 min
时间并集：[420, 600] ∪ [430, 580] = [420, 600] → 长度180 min

Temporal_IoU = 150 / 180 = 0.833
```

#### 步骤6: 计算综合IoU

有两种综合方式：

**方法A：乘法（更严格）**
```
Combined_IoU = Spatial_IoU × Temporal_IoU
             = 0.65 × 0.833
             = 0.541
```

**方法B：加权平均（可调）**
```
Combined_IoU = α × Spatial_IoU + β × Temporal_IoU
             (通常α=0.5, β=0.5)
             = 0.5 × 0.65 + 0.5 × 0.833
             = 0.742
```

### 数学表达

对于一维区间集合I₁和I₂：

```
1D_IoU(I₁, I₂) = Length(I₁ ∩ I₂) / Length(I₁ ∪ I₂)
```

分层IoU：
```
Spatial_IoU = 1D_IoU(KP区间₁, KP区间₂)
Temporal_IoU = 1D_IoU(Time区间₁, Time区间₂)
Combined_IoU = Spatial_IoU × Temporal_IoU  (或加权平均)
```

### 完整计算示例

**预测**：
- 热点1：KP [29, 35], Time [480, 600]
- 热点2：KP [4, 8], Time [420, 480]

**真实**：
- 拥堵1：KP [30, 34], Time [490, 580]
- 拥堵2：KP [31, 33], Time [500, 560]
- 拥堵3：KP [4.5, 7], Time [430, 470]

**空间维度**：
```
Pred_KP: [[4, 8], [29, 35]]
GT_KP:   [[4.5, 7], [30, 34]]

交集：
  [4, 8] ∩ [4.5, 7] = [4.5, 7] → 2.5 km
  [29, 35] ∩ [30, 34] = [30, 34] → 4 km
  总计：6.5 km

并集：
  [[4, 8], [29, 35]] → 10 km

Spatial_IoU = 6.5 / 10 = 0.65
```

**时间维度**：
```
Pred_Time: [[420, 600]]  (合并后)
GT_Time:   [[430, 580]]  (合并后)

交集：
  [420, 600] ∩ [430, 580] = [430, 580] → 150 min

并集：
  [420, 600] ∪ [430, 580] = [420, 600] → 180 min

Temporal_IoU = 150 / 180 = 0.833
```

**综合IoU**：
```
Combined_IoU (乘法) = 0.65 × 0.833 = 0.541
Combined_IoU (平均) = (0.65 + 0.833) / 2 = 0.742
```

### 诊断分析

| Spatial_IoU | Temporal_IoU | 诊断 | 建议 |
|-------------|--------------|------|------|
| 高 (>0.7) | 高 (>0.7) | 整体预测良好 | 继续保持 ✅ |
| 高 (>0.7) | 低 (<0.5) | 位置对，时间错 | 优化时间预测模型 ⚠️ |
| 低 (<0.5) | 高 (>0.7) | 时间对，位置错 | 优化空间预测模型 ⚠️ |
| 低 (<0.5) | 低 (<0.5) | 整体预测失败 | 重新设计模型 ❌ |

### 优点
- ✅ 可以独立诊断空间和时间预测的准确性
- ✅ 计算简单，无需复杂几何库
- ✅ 易于解释，便于模型调优
- ✅ 对一维投影的微小偏差更鲁棒

### 缺点
- ⚠️ **丢失了时空耦合信息**（最大缺点）
  - 例如："8点在KP30" vs "18点在KP30" → 空间IoU=1，时间IoU可能很高，但实际是不同的拥堵
- ⚠️ 不适合作为主要评估指标，只能作为补充
- ⚠️ 乘法综合可能过于严格，平均综合可能过于宽松

### 时空耦合丢失示例

**场景1**：
```
预测：早高峰KP30拥堵 (8:00, KP30)
真实：晚高峰KP30拥堵 (18:00, KP30)

Spatial_IoU = 1.0  (完全重合)
Temporal_IoU = 0.0  (完全不重合)
Combined_IoU = 0.0

多边形IoU ≈ 0.0  (正确反映：这是两个不同的拥堵)
```

**场景2**：
```
预测：早高峰KP30拥堵 (8:00, KP30)
真实：早高峰KP50拥堵 (8:00, KP50)

Spatial_IoU = 0.0  (完全不重合)
Temporal_IoU = 1.0  (完全重合)
Combined_IoU = 0.0

多边形IoU ≈ 0.0  (正确反映：位置错误)
```

这两种情况分层IoU都是0，但原因不同，需要查看Spatial和Temporal分别的值来诊断。

### 实现技术

```python
def get_kp_interval(polygon):
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    return (bounds[0], bounds[2])

def get_time_interval(polygon):
    bounds = polygon.bounds
    return (bounds[1], bounds[3])

def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def interval_iou(intervals1, intervals2):
    union1 = merge_intervals(intervals1)
    union2 = merge_intervals(intervals2)

    # 计算交集长度
    intersection_length = 0
    for i1 in union1:
        for i2 in union2:
            start = max(i1[0], i2[0])
            end = min(i1[1], i2[1])
            if end > start:
                intersection_length += (end - start)

    # 计算并集长度
    all_intervals = union1 + union2
    merged_all = merge_intervals(all_intervals)
    union_length = sum(end - start for start, end in merged_all)

    return intersection_length / union_length if union_length > 0 else 0

# 使用
spatial_iou = interval_iou(pred_kp_intervals, gt_kp_intervals)
temporal_iou = interval_iou(pred_time_intervals, gt_time_intervals)
combined_iou = spatial_iou * temporal_iou
```

---

## 三种指标对比总结

| 维度 | 多边形IoU | 栅格F1-Score | 分层IoU |
|------|-----------|--------------|---------|
| **计算复杂度** | 高 | 中 | 低 |
| **几何精确性** | 高 | 中 | 低 |
| **可解释性** | 高 | 高 | 高 |
| **诊断能力** | 低 | 高 | 高 |
| **时空耦合** | 保留 | 保留 | **丢失** ⚠️ |
| **鲁棒性** | 低 | 中 | 高 |
| **适用场景** | 整体评估 | 误报/漏报诊断 | 空间/时间分离诊断 |
| **推荐权重** | 主要指标 | 主要指标 | 辅助指标 |

### 综合使用建议

1. **主要参考**：多边形IoU 和 栅格F1-Score
2. **辅助诊断**：分层IoU（看具体是空间还是时间的问题）
3. **不同阶段**：
   - 模型开发：三个指标都看
   - 最终评估：多边形IoU为主，F1为辅
   - 问题调试：分层IoU定位问题

### 阈值参考

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|------|------|------|------|------|
| 多边形IoU | >0.7 | 0.5-0.7 | 0.3-0.5 | <0.3 |
| F1-Score | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| 空间IoU | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| 时间IoU | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |

---

## 实际应用案例

### 案例1: 预测过度

```
结果：
  多边形IoU: 0.45
  Precision: 0.55
  Recall: 0.95
  F1: 0.69
  Spatial_IoU: 0.60
  Temporal_IoU: 0.70

分析：
  - Recall很高但Precision低 → 预测过度，误报多
  - 多边形IoU中等 → 覆盖了真实区域但预测面积过大
  - 分层IoU中等 → 空间和时间范围都略微过大

建议：
  - 收紧预测边界
  - 提高热点筛选阈值
```

### 案例2: 预测不足

```
结果：
  多边形IoU: 0.35
  Precision: 0.85
  Recall: 0.50
  F1: 0.63
  Spatial_IoU: 0.80
  Temporal_IoU: 0.45

分析：
  - Precision高但Recall低 → 预测保守，漏报多
  - 空间IoU高但时间IoU低 → 位置预测准确，时间范围不够

建议：
  - 扩大时间预测范围
  - 降低热点筛选阈值
  - 检查时间聚类参数
```

### 案例3: 位置偏移

```
结果：
  多边形IoU: 0.20
  Precision: 0.40
  Recall: 0.45
  F1: 0.42
  Spatial_IoU: 0.30
  Temporal_IoU: 0.75

分析：
  - 所有指标都低，但时间IoU相对高 → 位置预测有偏差
  - 空间IoU低 → KP范围预测错误

建议：
  - 检查KP坐标计算逻辑
  - 验证数据预处理是否正确
  - 检查训练数据的KP分布
```

---

## 参考文献

1. **IoU (Intersection over Union)**
   - Jaccard Index: 集合相似度度量
   - 目标检测: YOLO, Faster R-CNN等使用

2. **F1-Score**
   - 信息检索: Precision & Recall
   - 机器学习: 不平衡数据集评估

3. **时空数据分析**
   - Spatio-temporal clustering
   - Traffic prediction and evaluation

---

*评估指标详解文档 - 2025年10月*
