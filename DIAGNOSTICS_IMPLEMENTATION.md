# 计算中心统一日志与自动诊断

## 边界

本改造不改变计算任务的 `network=deny` 边界，不给计算运行时增加外部网络、模型、Secret 或工具。`Workflow Diagnostic Sweep` 是独立的 GitHub Actions 运维工作流，只读取本仓库的 GitHub Actions 元数据和失败日志。

## 自动诊断

每 30 分钟扫描近期 Run。对所有运行记录 Run、Attempt、Commit SHA、Workflow、Job、Step、触发者和耗时；对失败、取消、超时和启动失败的运行额外下载完整日志，并完成凭据脱敏、关键错误抽取、失败 Step 定位、错误分类、失败指纹和重试建议。

识别类别包括：权限或 Secret、限流、超时、网络、依赖安装、Schema 或输入、Artifact 或证明、Provider 或模型、测试断言、资源耗尽、运行时异常和未知错误。

## 与计算业务诊断的关系

已有 `compute-diagnostics.json`、`compute-error.json`、`compute-audit.json`、`compute-console.log` 和 `artifact-manifest.json` 继续作为计算任务的业务权威证据。统一扫描器负责跨工作流索引与 GitHub Actions 外层故障诊断，不替代数值有效性、校准、收敛、敏感性、约束可行性或业务质量门。

## 诊断包

```text
summary.md
→ diagnostic-index.json
→ runs/<run_id>/failure.json
→ runs/<run_id>/key-lines.jsonl
→ runs/<run_id>/jobs.jsonl
→ runs/<run_id>/redacted-logs/
→ manifest.json
→ GitHub Artifact Attestation
```

诊断包保存 30 天。Manifest 对每个文件记录字节数和 SHA-256。Pull Request 验证阶段不生成 Attestation；合并后的定时或手动运行生成来源证明。

禁止记录完整环境变量、Authorization、Cookie、Token、API Key、SendKey、密码、原始敏感输入或业务 Secret。
