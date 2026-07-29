# 恢复与回滚

1. 停止创建新的 `[compute]` 和 `[literature]` 票据。
2. 核验失败 Run 的业务状态、Artifact、Manifest、SHA 和依赖矩阵。
3. 优先回滚到本仓库最近一个通过完整质量门的提交。
4. 只有发生仓库级结构损坏时，才使用 `MIGRATION_PROVENANCE.json` 中的固定迁移源重建。
5. 不从另外两个业务中心复制运行文件或 Secret；Secret 只按本仓库名称重新配置。
6. 恢复后必须重跑网络隔离、全能力、机构质量、依赖矩阵和文献边界验收。
