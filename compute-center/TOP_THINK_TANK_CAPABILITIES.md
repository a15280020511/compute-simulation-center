# 顶级智库计算能力扩展

本扩展在既有 26 项顶层操作和三中心隔离架构内，为 `finance_decision_analysis` 增加 38 个固定模式。所有模式均为只读计算、运行时断网、禁止票据代码、禁止动态安装、禁止券商交易，并在正式基准通过前保持 `controlled-preview`。

## 能力包

| 能力包 | 固定依赖 | 模式数量 | 主要用途 |
|---|---|---:|---|
| 数据工程 | DuckDB、Polars、PyArrow、Pandera、Pint | 3 | 大型证据表、连接、质量与单位审计 |
| 计量与经营 | Statsmodels、linearmodels、samplics、lifelines、ruptures、scikit-learn、pymrio | 21 | 政策、调查、商业、消费、运营和产业研究 |
| 金融风险 | Riskfolio-Lib、FinanceToolkit、Statsmodels | 4 | CVaR、回撤约束、财务诊断与因子归因 |
| 决策优化 | pymoo、Optuna、Pyomo、HiGHS | 6 | 多目标决策、固定搜索、沙盘和政策微观模拟 |
| 层次贝叶斯 | PyMC、ArviZ | 1 | 群体差异、部分汇聚、MCMC诊断 |
| 栅格与空间 | Xarray、Rasterio、rioxarray、libpysal、esda | 3 | 栅格统计、变化检测和空间自相关 |

## 新增模式

### 数据和证据工程

- `bounded_table_profile`
- `bounded_table_join`
- `schema_unit_validation`

### 计量、政策和调查

- `robust_glm`
- `panel_fixed_effects`
- `survey_weighted_estimation`
- `meta_analysis`
- `survival_analysis`
- `change_point_detection`
- `mixed_effects_model`
- `quantile_regression`
- `granger_causality`
- `power_analysis`
- `unobserved_components_forecast`
- `markov_regime_model`

### 商业、消费和经营

- `price_elasticity`
- `customer_lifetime_value`
- `customer_segmentation`
- `churn_probability`
- `marketing_mix_regression`
- `inventory_policy`
- `input_output_shock`
- `consumer_choice_logit`
- `process_capability`

### 金融和量化研究

- `cvar_portfolio`
- `drawdown_constrained_portfolio`
- `financial_ratio_analysis`
- `factor_attribution`

### 推理、决策和沙盘

- `multiobjective_pareto`
- `bounded_hyperparameter_search`
- `algebraic_resource_optimization`
- `strategic_sandbox`
- `influence_diagram`
- `policy_microsimulation`

### 贝叶斯和空间

- `hierarchical_bayesian_mean`
- `raster_zonal_statistics`
- `raster_change_detection`
- `spatial_autocorrelation`

## 安全边界

```text
数值运行网络：deny
模型调用：0
外部数据读取：0
票据提交代码：禁止
票据选择依赖：禁止
券商和交易执行：禁止
保证收益：禁止
个人心理诊断：禁止
自动政策或应急指令：禁止
```

## 晋级要求

每项模式从 `controlled-preview` 晋级生产必须同时通过：

1. 精确依赖安装与 `pip check`；
2. 固定真值、参数恢复或解析结果校验；
3. 边界、错误输入和资源上限测试；
4. 断网、无模型调用和无任意代码证明；
5. 冻结现实案例或样本外验证；
6. 适用假设、失效条件和回滚元数据审计。

本扩展不引入后台服务、共享数据库、动态路由或跨中心直连。
