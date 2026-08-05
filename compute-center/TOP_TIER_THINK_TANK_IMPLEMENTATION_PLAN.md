# 计算中心：顶级智库工具与方法实施方案

治理权威：`decision-system-governance/PUBLIC_INTELLIGENCE_AND_THINK_TANK_BLUEPRINT.json`。

## 定位

计算中心只接收治理中心批准的结构化证据包，在断网环境中完成可复算的研究、估计、预测、仿真、优化和政策分析。它不负责网页搜索、外部采集、实时流、任意代码或动态依赖。

## 五个机器目录

```text
registries/compute_tool_candidates.jsonl
registries/approved_compute_tools.json
registries/methods_and_standards.jsonl
benchmarks/frozen_benchmark_registry.json
benchmarks/tool_competition_results.jsonl
```

候选目录可以广；批准目录必须少、固定版本、固定依赖、固定哈希。

## 能力矩阵

### 1. 研究设计与证据方法

- 问题框架、理论机制和逻辑模型；
- 假设、估计量、测量和指标设计；
- 抽样、功效、实验和准实验设计；
- 系统综述、元分析和证据质量；
- 数据缺失、选择偏差和测量误差。

### 2. 统计与计量经济

- 描述统计、分布、检验和稳健估计；
- 线性、广义、混合和层级模型；
- 面板、空间、时间序列和状态空间；
- 生存、持续时间、计数和事件历史；
- 结构模型、宏观和行业模型。

候选生态包括 statsmodels、scikit-learn、linearmodels、arch、pmdarima、sktime 等，但必须通过基准筛选，不能因为知名直接晋级。

### 3. 因果推断

- 因果图、识别和可检验假设；
- 随机实验、匹配、加权和双重稳健；
- 差分中的差分、断点、工具变量和合成控制；
- 异质性效应、机制和中介；
- 安慰剂、反证、敏感性和未观测混杂分析。

重点候选：DoWhy/PyWhy、EconML、CausalML、DoubleML、SyntheticControlMethods 等。

### 4. 贝叶斯与概率编程

- PyMC、Stan、NumPyro、Pyro 等候选；
- 层级模型、隐变量、贝叶斯网络和决策网络；
- 先验预测、后验预测、诊断和校准；
- MCMC、变分推断、序贯蒙特卡洛；
- 价值信息和贝叶斯决策。

### 5. 预测和早期预警

- 经典、机器学习和组合预测；
- 概率预测、区间和分位数；
- 回测、滚动验证和评分规则；
- 变点、异常和制度切换；
- 多源领先指标和预警阈值。

重点候选：statsmodels、sktime、Darts、StatsForecast、MLForecast、PyOD、ruptures 等。

### 6. 不确定性、敏感性和可靠性

- Monte Carlo、Quasi-Monte Carlo 和重要性抽样；
- Sobol、Morris、FAST、HSIC 和局部敏感性；
- 不确定性传播、代理模型、PCE 和高斯过程；
- 可靠性、极值、稀有事件和风险；
- 参数、结构和情景不确定性。

重点候选：OpenTURNS、SALib、Chaospy、UQpy、scikit-optimize 等。

### 7. 优化和运筹

- 线性、整数、混合整数、凸和非线性；
- 约束规划、排程、路由、装箱和网络流；
- 多目标、稳健、随机和分布鲁棒优化；
- 资源配置、组合选择和政策组合；
- 对偶、敏感性、事后解释和情景重算。

重点候选：OR-Tools、CVXPY、Pyomo、PuLP、HiGHS、OSQP、SCS、Clarabel。

### 8. 仿真和复杂系统

- SimPy 类离散事件；
- Mesa 类 Agent-based modeling；
- PySD 类系统动力学；
- 微观模拟、人口合成和队列模型；
- 混合仿真、事件驱动和网络仿真；
- VV&A、参数校准、结构验证和可重复运行。

### 9. 战略、情景和政策分析

- CIA 公开结构化分析方法；
- ODNI/PHIA 来源、客观性、替代解释和不确定性标准；
- Delphi、地平线扫描、弱信号、驱动因素和三地平线；
- 2x2、形态分析、情景原型和情景发现；
- 政策压力测试、风洞、回溯和路线图；
- RAND 式稳健决策、深度不确定性和价值信息；
- 多准则决策、组合排序和权衡分析；
- 博弈、谈判、竞争、联盟和对手适应模型。

方法资料可进入方法目录；只有可验证的计算实现才进入工具候选目录。

### 10. 网络、关系和扩散

- NetworkX、igraph、graph-tool 等候选；
- 中心性、社群、结构洞、同配和鲁棒性；
- 多层、时序、二部和超图；
- 扩散、传染、级联和阈值模型；
- 企业所有权、供应链、贸易和技术网络。

### 11. 地理空间和遥感计算

- GeoPandas、Shapely、PyProj、Rasterio、xarray、rioxarray、odc-stac；
- 矢量、栅格、可达性、网络和空间统计；
- 遥感时间序列、土地利用和环境变化；
- 空间误差、尺度效应和情景地图。

### 12. 经济、金融和商业

- 宏观、行业、投入产出和社会核算；
- 成本效益、成本效果、真实期权和价值信息；
- 投资组合、因子、风险、压力测试和情景；
- 市场结构、竞争、需求、定价和消费者行为；
- 企业经营、供应链、库存、选址和资源配置。

### 13. 社会、心理和群体行为

- 调查和量表分析；
- 离散选择、潜类别和结构方程候选；
- 群体扩散、规范、极化和意见动力学；
- Agent、网络和空间行为模型；
- 模型输出不得用于秘密个人画像、操纵或自动执法。

### 14. 可复现性和质量

- 冻结输入、基准真值和版本；
- 锁文件、哈希、SBOM 和构建证据；
- 单元、属性、变形、数值稳定和压力测试；
- 确定性随机种子和资源上限；
- 跨模型一致性、校准、误差分解和失败诊断；
- Artifact、报告和独立复算。

## 全球候选发现

情报中心负责从以下生态发现候选并形成票据：PyPI、conda-forge、CRAN/R-universe、Julia General、GitHub/GitLab、OpenML、Hugging Face、Zenodo、Software Heritage、学术论文与官方文档。

计算中心不得自行联网搜索。候选票据必须含：

```text
name
capability_family
official_repository
package_registry
version
commit_or_release
license
hashes
dependency_graph
sbom
security_evidence
maintenance_evidence
runtime_requirements
network_requirement
benchmark_claims
existing_incumbent
expected_replacement_value
```

## 晋级竞争

```text
静态元数据与许可审计
→ 依赖和漏洞审计
→ 隔离构建
→ 单元与最小功能测试
→ 冻结基准
→ 数值稳定和压力测试
→ 与现有工具并行比较
→ 复算与 Artifact 审计
→ 治理批准
→ 固定版本、哈希和依赖
→ Canary
→ 生产晋级
```

晋级条件：补齐缺失能力，或在质量、安全、可复现、资源、维护或基准上显著优于现有工具。功能重叠但没有显著增益的候选应拒绝。

## 基准设计

每个能力族至少包含：

- 小型确定性真值；
- 真实公开数据冻结样本；
- 缺失、异常、尺度和边界条件；
- 大小数据和资源压力；
- 失败输入和错误诊断；
- 运行时间、内存、精度、稳定性和可解释性；
- 跨版本回归和跨工具一致性。

## 调度

- 每月：生产工具漏洞、依赖、兼容性和基准回归；
- 每季度：新候选与现有工具竞争，评估替代和简化；
- 每年：能力矩阵、方法标准和冻结基准全面复核；
- 严重漏洞或上游停维：立即进入降级评审，但不得自动删除生产工具。
