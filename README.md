# Computation & Simulation Center

独立的计算、校准、仿真、优化和不确定性量化中心。网页 GPTs 不得直接控制本仓库；`a15280020511/decision-system-governance` 是唯一外部控制、任务转交和证据中继入口。

## 正式入口

以下 Issue 前缀只作为治理仓库创建子任务时使用，不是 GPTs 的直接入口：

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

## 计算基准库

私有 Hugging Face Dataset `compute-numeric-baselines` 是本中心的外部纯数值基准库，但本中心不是存储网关，也不得联网读取它。

```text
情报中心生成纯数值 Artifact
→ 治理仓库核验并写入私有基准库
→ 治理仓库按具体任务生成不可变转交包
→ 计算中心在断网执行前取得转交包
→ network=deny 下计算
```

本仓库不得配置 `HF_TOKEN`，不得直接访问 Hugging Face，不得直接读取情报中心 Artifact。基准包必须由治理仓库按任务转交，并绑定 Manifest、SHA 和来源。

## 隔离边界

- 不调用数据证据中心或专家研判中心；
- 不使用跨仓库 `repository_dispatch`、运行时 Artifact 下载或共享业务 Secret；
- 不接受 GPTs 直接控制；
- 不持有私有 Dataset 凭据；
- 数值任务不联网、不调用模型；
- 结果必须核验业务状态、Artifact、Manifest 和 SHA，不能只看 Workflow success。

## 维护入口

- 本地运行：`OPERATIONS_RUNBOOK.md`
- 故障恢复：`RECOVERY.md`
- 安全边界：`SECURITY.md`
- 迁移溯源：`MIGRATION_PROVENANCE.json`
- 治理兼容：`governance-compatibility.json`
