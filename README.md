# Computation & Simulation Center

本仓库由 `a15280020511/test` 在固定提交 `abac3d776340c8c162b8fc0c670167fde94f3baa` 拆分迁入。

## 职责

统计计算、参数校准、假设治理、蒙特卡罗、仿真、因果分析、优化、不确定性量化与真实结果反馈。

## 机器权威目录

`compute-center/compute-capabilities.json`

## 隔离边界

- 本仓库只运行本中心任务。
- GPTs 是三个业务中心之间唯一的控制与证据中继。
- 禁止中心间直接调用、运行时导入、Artifact 互取和共享业务 Secret。
- 原业务目录 `compute-center/` 暂时保留，避免迁移与路径重构同时发生。
- 旧仓库在验收完成前保留为治理记录和回滚源，本次不删除旧内容。

## 迁移证据

查看 `MIGRATION_MANIFEST.json`、`MIGRATION.md` 和 `governance-compatibility.json`。

## V3 accuracy enhancement

See `compute-center/accuracy-enhancement-capabilities.json`, `compute-center/ACCURACY_ENHANCEMENT_STATUS.md`, the institutional quality matrix, and the separate allowlisted literature-evidence workflow. New modes remain controlled-preview until domain benchmarks pass.
