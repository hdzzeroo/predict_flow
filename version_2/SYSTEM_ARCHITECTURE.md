# 多Agent交通预测系统 - 运行时架构

## 系统运行时结构图

```mermaid
graph TB
    %% 用户输入层
    User["👤 用户<br/>输入: 数据文件 + 预测需求"]

    %% 主系统层
    Main["🎯 TrafficPredictionSystem<br/>主控制器"]

    %% LLM客户端
    LLM["🤖 LLM Client<br/>模型接口层"]

    %% Agent层
    subgraph Agents["📦 Agent层"]
        HDA["📊 HybridDataAgent<br/>混合数据分析"]
        PRA["🔍 PatternRecognitionAgent<br/>模式识别"]
        PEA["🎯 PredictionExpertAgent<br/>预测专家<br/>(未完全实现)"]
    end

    %% 数据处理层
    subgraph DataProcessing["⚙️ 数据处理层"]
        DL["📁 DataLoader<br/>数据加载"]
        TG["🔺 TriangleGenerator<br/>三角形生成"]
        PM["📝 PromptManager<br/>提示词管理"]
    end

    %% 算法层
    subgraph Algorithms["🧮 算法层"]
        DBSCAN["📍 DBSCAN聚类"]
        OD["🔄 重叠检测"]
        VIS["📈 可视化生成"]
    end

    %% 输出层
    Output["📋 预测结果<br/>JSON报告 + 可视化"]

    %% 连接关系
    User --> Main
    Main --> HDA
    Main --> PRA
    Main --> Output

    HDA --> DL
    HDA --> TG
    HDA --> PM
    HDA --> LLM

    PRA --> DBSCAN
    PRA --> OD
    PRA --> PM
    PRA --> LLM

    TG --> VIS

    LLM -.-> |API调用| External["☁️ 外部LLM服务<br/>OpenAI/Claude/Mock"]

    style User fill:#e1f5fe
    style Main fill:#fff3e0
    style HDA fill:#f3e5f5
    style PRA fill:#f3e5f5
    style PEA fill:#ffebee,stroke-dasharray: 5 5
    style LLM fill:#e8f5e9
    style Output fill:#fce4ec
```

## 详细运行时流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as Main System
    participant HD as HybridDataAgent
    participant PR as PatternAgent
    participant LLM as LLM Client
    participant O as Output

    U->>M: 1. 提供数据文件和预测需求
    activate M

    M->>HD: 2. 执行数据分析
    activate HD
    HD->>HD: 2.1 加载原始数据
    HD->>HD: 2.2 生成三角形
    HD->>LLM: 2.3 请求数据洞察
    LLM-->>HD: 返回分析结果
    HD->>HD: 2.4 统计分析
    HD->>HD: 2.5 异常检测
    HD-->>M: 返回分析结果
    deactivate HD

    M->>PR: 3. 执行模式识别
    activate PR
    PR->>PR: 3.1 检测重叠
    PR->>PR: 3.2 DBSCAN聚类
    PR->>PR: 3.3 时空模式分析
    PR->>LLM: 3.4 请求模式增强
    LLM-->>PR: 返回模式洞察
    PR-->>M: 返回模式结果
    deactivate PR

    M->>M: 4. 生成预测报告
    M->>M: 4.1 基于重叠预测
    M->>M: 4.2 生成可视化
    M->>M: 4.3 计算置信度

    M->>O: 5. 输出结果
    O-->>U: 返回预测报告
    deactivate M
```

## 核心数据流

```mermaid
graph LR
    subgraph Input["输入数据"]
        CSV["📄 CSV文件<br/>date, 発生Ｋｐ<br/>渋滞時間, ピーク長"]
    end

    subgraph Stage1["阶段1: 数据转换"]
        RAW["原始数据<br/>DataFrame"]
        TRI["🔺 三角形数据<br/>List[Dict]"]
    end

    subgraph Stage2["阶段2: 模式识别"]
        OVL["重叠区域<br/>55个检测"]
        CLU["聚类结果<br/>2个聚类"]
        PAT["模式特征<br/>时空规律"]
    end

    subgraph Stage3["阶段3: 预测生成"]
        PRED["预测结果<br/>KP 13.00<br/>置信度 100%"]
        VIZ["可视化<br/>三角形分布图"]
    end

    CSV --> RAW
    RAW --> TRI
    TRI --> OVL
    TRI --> CLU
    OVL --> PAT
    CLU --> PAT
    PAT --> PRED
    TRI --> VIZ
```

## 关键组件交互

### 1. **数据处理流程**
```
原始CSV → DataLoader → DataFrame → TriangleGenerator → 三角形抽象
         ↓
      60条记录 → 预处理 → 时间窗口分组 → 16个三角形
```

### 2. **三角形结构**
```python
Triangle = {
    'id': 'triangle_2023-11-01_0830_130',
    'vertices': {
        'apex': [0.5, 13.0],      # [时间偏移, KP位置]
        'base_left': [0.0, 12.5],  # 起始点
        'base_right': [1.5, 13.5]  # 结束点
    },
    'properties': {
        'area': 0.75,              # 三角形面积
        'intensity_score': 0.85,   # 强度分数
        'duration_hours': 1.5      # 持续时间
    }
}
```

### 3. **重叠检测机制**
```
Triangle_A ∩ Triangle_B → 重叠区域
                        ↓
              计算重叠比例 > 0.3 → 记录为有效重叠
                        ↓
                 55个重叠 → 预测依据
```

### 4. **预测逻辑**
```
最密集重叠区域 → 提取中心位置 → KP 13.00
     ↓
重叠比例 → 置信度计算 → 100%
     ↓
重叠数量 > 50 → 风险级别 → HIGH
```

## 系统特性

| 组件 | 功能 | 状态 | 关键输出 |
|------|------|------|----------|
| **HybridDataAgent** | 数据分析+三角形生成 | ✅ 完整实现 | 16个三角形 |
| **PatternRecognitionAgent** | 模式识别+聚类 | ✅ 完整实现 | 55个重叠，2个聚类 |
| **PredictionExpertAgent** | 高级预测（多模态） | ⚠️ 部分实现 | 基础预测在Main中 |
| **LLM Client** | 模型接口 | ✅ 支持Mock/OpenAI | 数据洞察，模式分析 |
| **TriangleGenerator** | 三角形生成 | ✅ 核心功能 | 时空抽象表示 |

## 运行时配置

```yaml
系统模式:
  - 演示模式: 使用Mock LLM，无需API密钥
  - 生产模式: 使用真实LLM API

数据规模:
  - 快速演示: 60条记录 → 16个三角形
  - 完整演示: 11456条记录 → 5226个三角形

关键参数:
  - min_duration: 10分钟（最小拥堵时间）
  - min_length: 0.5km（最小拥堵长度）
  - overlap_threshold: 0.3（重叠阈值）
  - clustering_eps: 1.5（DBSCAN参数）
```

## 错误处理机制

```mermaid
graph TD
    A[Agent执行] --> B{是否出错?}
    B -->|否| C[返回结果]
    B -->|是| D[记录错误日志]
    D --> E[抛出RuntimeError]
    E --> F[向上传播]
    F --> G[系统级处理]
    G --> H[保存错误状态]
    H --> I[返回错误报告]
```

## 性能指标

- **数据处理速度**: ~200条/秒
- **三角形生成**: ~100个/秒
- **重叠检测**: O(n²) 复杂度
- **聚类分析**: DBSCAN O(n log n)
- **总体延迟**: 小数据集 < 1秒，大数据集 ~30秒