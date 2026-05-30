# LLM热点识别Prompt改进总结

## 📅 更新时间
2025-10-28

## 🎯 改进目标
1. 使用XML结构化Prompt，提高可读性和层次性
2. 设计4步思维链（CoT），强制LLM进行详细推理
3. 明确固定参数，消除模糊性
4. 压缩Prompt长度，减少Token消耗
5. 升级到gpt-4o，提升推理能力
6. 添加数量控制提示，避免预测过少

## ✅ 已完成的改进

### 1. System Prompt重写
**之前**（4行）:
```
你是一个资深的交通拥堵预测专家，拥有多年的交通数据分析经验。
你的任务是基于历史数据预测未来可能发生拥堵的区域。
你需要深度思考每个拥堵事件之间的关联性...
```

**现在**（9行）:
```
你是资深交通拥堵预测专家，擅长基于历史数据分析预测未来拥堵区域。

请使用结构化思维链方式分析数据：
1. 先进行空间-时间聚类
2. 分析每个聚类的几何边界
3. 计算预测形状的顶点坐标
4. 验证结果的合理性

输出必须是严格的JSON格式，包含thinking字段记录你的推理过程。
```

**改进点**:
- ✅ 明确4步思维链
- ✅ 要求输出thinking字段
- ✅ 更简洁清晰

---

### 2. User Prompt完全重写（使用XML结构）

**之前**: 360行，32KB
**现在**: 约260行，17KB
**减少**: 47%

**新结构**:
```xml
<task>任务描述</task>

<context>
  <data_source>数据来源信息</data_source>
  <data_characteristics>数据特征</data_characteristics>
</context>

<methodology>
  <core_principle>核心原理</core_principle>
  <clustering_parameters>
    <spatial_threshold>7.0 km</spatial_threshold>  <!-- 固定值！-->
    <temporal_threshold>180 minutes</temporal_threshold>  <!-- 固定值！-->
  </clustering_parameters>
  <shape_decision_rules>形状决策规则</shape_decision_rules>
</methodology>

<input_data>完整的三角形JSON数据</input_data>

<thinking_instructions>
  <step_1_clustering>详细的聚类指令</step_1_clustering>
  <step_2_boundary_analysis>详细的边界分析指令</step_2_boundary_analysis>
  <step_3_shape_construction>详细的顶点计算指令</step_3_shape_construction>
  <step_4_validation>详细的验证指令</step_4_validation>
</thinking_instructions>

<output_format>JSON格式示例（包含thinking字段）</output_format>

<important_guidelines>重要指导原则</important_guidelines>
```

**改进点**:
- ✅ XML标签明确层次结构
- ✅ 删除了重复的几何约束说明（减少3-4次重复）
- ✅ 删除了过多的物理原理解释
- ✅ 固定参数：KP=7km（不再是5-10km），Time=180min（不再是2-4小时）
- ✅ 添加预期热点数量提示：`estimated_hotspots = triangle_count * 0.25`

---

### 3. 4步思维链（CoT）设计

#### **Step 1: 空间-时间聚类**
```xml
<objective>识别空间-时间聚集的事件组</objective>
<method>
1. 计算两两距离（空间距离、时间距离）
2. 判断条件：空间≤7km 且 时间≤180分钟
3. 优先级：三年>两年>一年
4. 最小聚类：至少2个事件
</method>
<expected_output>
在thinking.step1_clustering中输出：
- 聚类列表
- 包含的三角形ID
- 聚类理由
- 年份覆盖情况
</expected_output>
```

**核心改进**:
- ✅ 明确计算方法
- ✅ 固定阈值（不再模糊）
- ✅ 要求输出详细推理

#### **Step 2: 边界分析（关键！）**
```xml
<objective>分析每个聚类的几何边界，决定使用三角形还是梯形</objective>
<method>
1. 识别"前端边界"（垂直底边位置）
   - 提取所有底边的KP值
   - 计算：front_kp_min, front_kp_max

2. 识别"后端边界"（顶点/峰值位置）
   - 提取所有峰值点的KP值
   - 计算：back_kp_min, back_kp_max

3. 分析时间分布
   - 计算时间跨度

4. 决策形状类型
   - 时间跨度 ≤ 120分钟 → 三角形
   - 时间跨度 > 120分钟 → 梯形
</method>
<expected_output>
在thinking.step2_boundaries中输出JSON格式：
{
  "cluster_1": {
    "front_kp_boundary": [30.0, 33.0],
    "back_kp_boundary": [42.0, 48.0],
    "time_span_minutes": 145,
    "shape_decision": "trapezoid",
    "reasoning": "..."
  }
}
</expected_output>
```

**核心改进**:
- ✅ 明确前端/后端边界的计算方法
- ✅ 固定形状决策阈值：120分钟
- ✅ 要求输出结构化的JSON

#### **Step 3: 顶点计算**
```xml
<objective>根据边界计算精确的vertices坐标</objective>
<method>
【三角形构造】：
vertex1 = [front_kp, 最早时间]  # 底边下端
vertex2 = [front_kp, 最晚时间]  # 底边上端（KP与vertex1相同）
vertex3 = [back_kp, 峰值平均时间]  # 顶点

验证：vertex1[0] == vertex2[0]（底边垂直）

【梯形构造】：
vertex1 = [front_kp, 前端最早时间]  # 左下
vertex2 = [back_kp, 后端最早时间]   # 右下（时间通常与vertex1不同）
vertex3 = [back_kp, 后端最晚时间]   # 右上
vertex4 = [front_kp, 前端最晚时间]  # 左上

验证：
- vertex1[0] == vertex4[0]（左边垂直）
- vertex2[0] == vertex3[0]（右边垂直）
- vertex1[1] != vertex2[1]（下边不水平）
- vertex3[1] != vertex4[1]（上边不水平）
</method>
```

**核心改进**:
- ✅ 详细的构造公式
- ✅ 明确的验证规则
- ✅ 避免生成矩形的约束

#### **Step 4: 自我验证**
```xml
<objective>验证分析结果的合理性</objective>
<checks>
1. 数量检查：预期热点数量 vs 实际识别数量
2. 覆盖率检查：应该 > 70%
3. 几何有效性：是否符合约束
4. 优先级检查：三年数据是否都识别了
</checks>
```

**核心改进**:
- ✅ 自我检查机制
- ✅ 发现问题可以自我修正
- ✅ 提高输出质量

---

### 4. 明确的固定参数

| 参数 | 之前 | 现在 | 改进 |
|------|------|------|------|
| KP聚类阈值 | `±5-10km`（模糊） | `7.0 km`（固定） | ✅ 消除歧义 |
| 时间聚类阈值 | `±2-4小时`（模糊） | `180 minutes (3 hours)`（固定） | ✅ 消除歧义 |
| 形状决策阈值 | 未明确 | `120 minutes (2 hours)`（固定） | ✅ 新增明确规则 |
| 预期热点数量 | 未提示 | `triangle_count × 0.25`（动态提示） | ✅ 避免预测过少 |

---

### 5. API参数优化

#### **config.py**
```python
# 之前
openai_model = "gpt-4o-mini"
openai_temperature = 0.1
openai_max_tokens = 16000
openai_timeout = 60

# 现在
openai_model = "gpt-4o"         # 升级到gpt-4o
openai_temperature = 0.2        # 提高创造性
openai_max_tokens = 8000        # 适配新Prompt（更短但更结构化）
openai_timeout = 60             # 保持60秒
```

#### **llm_analyzer.py**
默认值同步更新为：
- `model = "gpt-4o"`
- `temperature = 0.2`
- `max_tokens = 8000`
- `timeout = 60`

---

### 6. 新的输出格式（包含thinking字段）

```json
{
  "thinking": {
    "step1_clustering": "识别出4个聚类。Cluster 1包含三角形[0,3,5,8]...",
    "step2_boundaries": {
      "cluster_1": {
        "front_kp_boundary": [30.0, 33.0],
        "back_kp_boundary": [42.0, 48.0],
        "time_span_minutes": 145,
        "shape_decision": "trapezoid",
        "reasoning": "时间跨度145分钟>120分钟，使用梯形"
      }
    },
    "step3_construction": "Cluster 1（梯形）：前端KP=30.0...",
    "step4_validation": "验证结果：识别4个热点（预期4个）✓..."
  },
  "direction": "上",
  "hotspots": [...],
  "summary": {...}
}
```

**好处**:
- ✅ 可以追踪LLM的推理过程
- ✅ 便于调试和优化
- ✅ 提高可解释性

---

## 📊 改进效果预期

### 当前问题 vs 预期改进

| 问题 | 当前状态 | 预期改进 |
|------|---------|---------|
| **热点数量偏少** | 下行3个（实际6个），上行3个（实际15个） | 预期4-6个（下行），5-8个（上行） |
| **Spatial IoU低** | 下行0.08，上行0.45 | 预期0.3+，0.6+ |
| **F1 Score低** | 下行0.04，上行0.54 | 预期0.3+，0.7+ |
| **Recall低** | 下行0.05，上行0.69 | 预期0.5+，0.8+ |
| **过度合并** | LLM倾向于生成少量大热点 | 通过CoT和数量提示改善 |

### 改进机制

1. **明确参数** → 减少LLM的不确定性
2. **CoT强制推理** → 避免跳步思考
3. **数量提示** → 避免过度合并
4. **边界分析** → 更精确的形状计算
5. **自我验证** → 提高输出质量
6. **gpt-4o升级** → 更强的推理能力

---

## 📁 修改的文件

1. ✅ `prompt_templates.py` - 完全重写Prompt（360行→260行，减少47%）
2. ✅ `config.py` - 更新API参数（gpt-4o, temperature=0.2, max_tokens=8000）
3. ✅ `llm_analyzer.py` - 更新默认参数
4. ✅ `data_preparers.py` - 已添加数据规范化功能（之前完成）

---

## 🧪 测试建议

### 1. 单次测试
```bash
cd /home/dizhihuang/graduate/predict_workflow/version1_3
python test_complete_workflow.py

# 输入: 2025/5/4関越
# 观察：
# 1. thinking字段的详细程度
# 2. 热点数量是否增加
# 3. 形状是否符合几何约束
```

### 2. 批量评估
```bash
python batch_evaluation.py

# 对比：
# 1. Polygon IoU提升多少
# 2. Grid F1-Score提升多少
# 3. Spatial/Temporal IoU变化
```

### 3. 关注指标
- 热点数量：应该接近 `triangle_count × 0.25`
- 覆盖率：应该 > 70%
- Spatial IoU：目标 > 0.3（下行），> 0.6（上行）
- F1 Score：目标 > 0.3（下行），> 0.7（上行）

---

## 💡 后续优化方向（如果效果不佳）

### Plan B: 两阶段调用
如果CoT方案效果仍不理想，可以考虑：
1. 第一次调用：只做聚类
2. 第二次调用：根据聚类结果绘制形状

### Plan C: Few-shot示例
添加2-3个真实成功案例到Prompt中

### Plan D: 调整参数
- 进一步提高temperature（0.2 → 0.3）
- 调整预期热点比例（0.25 → 0.3）
- 修改阈值（7km → 10km）

---

## 📝 备注

1. 备份文件保存在：`prompt_templates.py.backup`（旧版本，32KB）
2. 新版本文件大小：17KB
3. Prompt长度减少：约100行（28%）
4. 所有改进已完成，等待测试验证

---

## ✅ 完成检查清单

- [x] 重写System Prompt（更简洁，明确CoT）
- [x] 重写User Prompt（XML结构化，260行）
- [x] 设计4步思维链（聚类→边界→顶点→验证）
- [x] 明确固定参数（7km, 180min, 120min）
- [x] 添加预期热点数量提示
- [x] 升级到gpt-4o
- [x] 调整API参数（temperature=0.2, max_tokens=8000）
- [x] 删除冗余内容（减少47%）
- [x] 添加thinking输出字段
- [x] 创建改进总结文档

---

**准备就绪，可以开始测试！** 🚀
