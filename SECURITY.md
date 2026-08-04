# 安全边界

- 不提交 Secret 值、令牌、私钥或个人数据。
- 网页 GPTs 不得直接控制本仓库；唯一外部控制者是 `a15280020511/decision-system-governance`。
- 不允许另一个业务中心读取本仓库运行目录、Environment Secret 或 Artifact。
- 不使用 Git submodule、跨仓库运行时 Artifact 下载或中心间 `repository_dispatch`。
- 不配置或使用 `HF_TOKEN`，不直接访问私有 Hugging Face Dataset。
- 计算基准数据只能由治理仓库以任务级不可变数据包转交；本仓库不得直接读取情报中心 Artifact。
- 数值执行保持 `network=deny`；不得为了读取基准库临时开放网络。
- 公共合同只能使用冻结副本、版本和哈希；业务运行时不得跨仓库读取治理文件。
