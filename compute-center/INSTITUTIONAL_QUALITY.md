# 计算中心机构级质量、校准与反馈闭环

## 目标

计算中心必须区分三个概念：

1. **算法成功**：固定算法完成运行，输入输出满足数值合同；
2. **结果可信**：通过数据、假设、真值、回测、校准和漂移检查；
3. **允许决策使用**：质量门明确给出 `DECISION_RELEASED`。

算法成功不自动等于结果可信，也不自动允许用于投资、公共政策或高风险决策。

## 默认规则

正式任务默认使用 `strict`：

- 正式任务最大假设变量比例：25%；
- 高风险任务最大假设变量比例：10%；
- 低置信度假设必须有区间或分布并经用户批准；
- 概率结论必须接受真实结果反馈校准；
- 高风险任务必须附带已通过的独立算法或求解器交叉验证；
- 严重失准或数据漂移时，旧模型禁止继续用于正式决策。

探索性任务可以显式设置 `quality_profile.decision_class=exploratory`，但其结果不得伪装为正式结论。

## GPTs与计算中心的职责

### GPTs负责

- 获取并核验观测数据；
- 提出、标记和解释假设；
- 保存决策时的预测、数据快照哈希、模型版本和适用范围；
- 在真实结果发生后，将预测与实际结果重新提交给计算中心；
- 根据质量报告修改假设、参数或数据，而不是覆盖旧结果。

### 计算中心负责

- 严格检查输入和假设比例；
- 运行固定算法；
- 计算Brier Score、ECE、Log Loss和预测区间覆盖率；
- 计算PSI、KS统计量和标准化均值漂移；
- 根据固定阈值决定结果是发布、附条件发布还是阻断；
- 保留旧结果、校准结果和哈希，形成不可变审计链。

计算中心仍然断网，不主动寻找反馈数据，也不直接调用API中心或专家团。

## 反馈票据示例

```json
{
  "task_id": "calibration-feedback-example-20260729",
  "objective": "用已发生的真实结果校准此前概率预测",
  "operation": "descriptive_statistics",
  "inputs": {"data": [0.1, 0.3, 0.7, 0.9]},
  "quality_profile": {
    "decision_class": "formal",
    "probabilistic_claim": true,
    "benchmark_ids": ["golden-descriptive-001"]
  },
  "calibration_feedback": {
    "predicted_probabilities": [0.1, 0.3, 0.7, 0.9],
    "observed_outcomes": [0, 0, 1, 1],
    "prediction_intervals": [
      {"lower": 80, "upper": 120, "actual": 110}
    ],
    "reference_values": [90, 95, 100, 105, 110],
    "recent_values": [94, 98, 102, 108, 112],
    "feedback_window": "真实结果发生后的固定观察窗口"
  },
  "data_context": {
    "variables": [
      {
        "name": "observed_outcomes",
        "required": true,
        "source_type": "user_provided",
        "confidence": "high",
        "sample_size": 4,
        "missing": false,
        "replacement_strategy": "none"
      }
    ]
  }
}
```

## 输出约束

每次成功计算新增：

- `compute-calibration.json`：校准和漂移指标；
- `compute-quality-report.json`：质量检查、发布状态和补救要求；
- `compute-result.json.quality`：供GPTs读取的紧凑质量结论；
- `compute-audit.json`：质量报告哈希和正式使用许可；
- `compute-summary.md`：人工可读的发布状态。

### 发布状态

- `DECISION_RELEASED`：允许在声明范围内用于正式决策；
- `DECISION_CONDITIONAL`：可以阅读和探索，但需要补充基准、反馈或其他证据；
- `DECISION_BLOCKED`：禁止用于正式决策，必须重新取数、校准或建模。

## 校准阈值

默认最低反馈样本为20。少于20时不会把校准结论视为稳定证据。

概率校准：

- Brier Score超过0.25警告，超过0.35阻断；
- ECE超过0.10警告，超过0.20阻断。

漂移：

- PSI超过0.10警告，超过0.25阻断；
- KS统计量超过0.15警告，超过0.30阻断；
- 标准化均值偏移超过0.5警告，超过1.0阻断。

阈值是统一治理起点。不同业务可通过正式版本升级修改，禁止由单张票据临时放宽阻断阈值。

## 基准体系

### 数值真值基准

`benchmarks/golden/manifest.json`包含解析公式、精确线性关系、标准WGS84距离、共轭贝叶斯后验和马尔可夫矩阵真值。任何必需真值失败，相关operation不得标记为decision-grade。

### 真实场景冻结基准

`benchmarks/frozen-real/manifest.json`固定使用`statsmodels==0.14.6`内置的Longley、Nile、Sunspots、Macrodata和Spector数据。每个数据集保存规范化SHA256，数据变化即阻断。

真实冻结基准用于检验：

- 病态回归诊断；
- 留出预测误差；
- 周期模式；
- 概率校准；
- 跨时期分布漂移。

## 能力边界

该体系提高的是可复现性、校准能力和决策纪律，不会自动补足：

- 私有机构数据库；
- 长期人工维护的数据资产；
- 现场调查和行业渠道；
- 领域专家的隐性知识；
- 法律授权、组织执行力和责任承担。

因此，计算中心可以达到机构级的确定性计算和质量控制能力，但不能单独等同于完整的顶级智库或国家级决策机构。
