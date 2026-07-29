# Computation & Simulation Center

独立的计算、校准、仿真、优化和不确定性量化中心。GPTs 是跨中心唯一控制与证据中继。

## 正式入口

- 数值计算 Issue：`[compute]`
- 文献证据 Issue：`[literature]` 或 `[compute-literature]`
- 机器权威能力目录：`compute-center/compute-capabilities.json`
- 准确度增强目录：`compute-center/accuracy-enhancement-capabilities.json`

## 运行面

```text
compute-numeric-offline
  └─ OS network namespace 强制断网

compute-literature-evidence
  └─ 仅 OpenAlex＋Crossref，输出冻结候选证据包
```

文献证据不能调用数值 Dispatcher，也不能把论文数值自动提升为正式参数。新增高级模式在领域基准完成前保持 `controlled-preview`。

## 隔离边界

- 不调用数据证据中心或专家研判中心；
- 不使用跨仓库 `repository_dispatch`、运行时 Artifact 下载或共享业务 Secret；
- 数值任务不联网、不调用模型；
- 结果必须核验业务状态、Artifact、Manifest 和 SHA，不能只看 Workflow success。

## 维护入口

- 本地运行：`OPERATIONS_RUNBOOK.md`
- 故障恢复：`RECOVERY.md`
- 安全边界：`SECURITY.md`
- 迁移溯源：`MIGRATION_PROVENANCE.json`
- 治理兼容：`governance-compatibility.json`
