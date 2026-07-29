# Computation & Simulation Center 运行手册

## 入口

- `[compute]`：断网数值计算。
- `[literature]` / `[compute-literature]`：OpenAlex＋Crossref 冻结候选证据。

## 权威工作流

- `compute-ticket.yml`：正式数值票据；
- `literature-evidence-ticket.yml`：正式文献票据；
- `compute-all-operations-validate.yml`：全部生产操作和决策智能模式；
- `compute-quality-benchmarks.yml`：机构级质量门；
- `institutional-quality-matrix.yml`：可选依赖隔离矩阵；
- `compute-network-isolation-validate.yml`：数值断网边界；
- `literature-evidence-validate.yml`：文献主机白名单和票据合同；
- `mesa-validate.yml`：Agent 仿真；
- `repository-line-audit.yml`：全仓静态审计。

## 纪律

1. 每个任务只安装对应受管依赖；CVXPY/HiGHS 与 OR-Tools 不在同一进程加载。
2. 正式输出以业务完成状态、正文、Artifact、Manifest 和 SHA 为准。
3. 新增模式先进入 `controlled-preview`，不得用工具已安装替代领域基准。
4. 依赖升级必须通过本仓库全部相关 CI；禁止运行时安装票据指定包。
5. 故障只做有限重试；重复任务使用 task ID、语义指纹和当前运行锁阻断。
