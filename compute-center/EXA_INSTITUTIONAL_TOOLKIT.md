# Exa 顶级智库工具扩展

本扩展由情报中心 Exa 进行两轮检索：第一轮按十一个专业领域发现候选，第二轮排除已发现项目后补齐第二梯队。计算中心只接收具备独立能力、明确安全边界、可在 Python 3.12 下按固定依赖安装、可在断网运行时执行的工具。

## 结果

```text
Exa 检索任务：20
检索结果槽位：200
入选工具/模式：41
运行时网络：deny
模型调用：0
外部数据获取：0
票据提交代码：禁止
票据选择依赖：禁止
自动交易：禁止
成熟度：controlled-preview
```

## 能力包

| 能力包 | 模式数 | 主要工具 |
|---|---:|---|
| 经济计量与因果 | 5 | pyfixest、DoubleML、EconML、semopy、PyBLP |
| 预测与极端风险 | 6 | StatsForecast、HierarchicalForecast、arch、PyOD、pyextremes、xskillscore |
| 决策与博弈 | 3 | EMA Workbench、pymcdm、nashpy |
| 空间与城市 | 7 | GeoPandas、mgwr、momepy、spreg、spopt、MovingPandas、segregation |
| 能源 | 2 | PyPSA、pandapower |
| 气候、供水与公共卫生 | 3 | WNTR、xclim、Starsim |
| 金融工程 | 2 | QuantLib、pyvinecopulib |
| 知识工程 | 4 | Splink、RapidFuzz、pySHACL/RDFLib、datasketch |
| 工程与运筹 | 5 | python-control、reliability、Stockpyl、Ciw、JobShopLib |
| 模型治理 | 4 | Fairlearn、Cleanlab、SHAP、Copulas |

## 治理原则

每个模式：

1. 只能从固定注册表进入。
2. 只能安装仓库精确锁定的能力包。
3. 不接受代码、公式脚本、FMU原生二进制或动态依赖。
4. 只读取票据中已经提供的结构化数据。
5. 必须输出模型版本、输入规模与有限数值。
6. 必须保留 `external_data_fetches=0`、`brokerage_execution=false` 和 `arbitrary_code_allowed=false`。
7. 在冻结现实基准、边界测试和结果反馈完成前不得提升为生产成熟度。

## 明确剔除

完整剔除清单位于 `institutional-tool-catalog.json`。主要原因包括：

- 与现有能力重复；
- 项目维护或许可证不足；
- 依赖外部数据库或国家法规包；
- 允许执行票据提供的模型代码或原生二进制；
- 主要价值来自实时联网、仪表盘或自动交易；
- 会把计算中心变成通用运行环境。
