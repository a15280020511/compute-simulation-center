# 文献证据运行面

此目录只负责生成冻结文献证据包，不调用数值 Dispatcher。网络仅允许 `api.openalex.org` 与 `api.crossref.org`。

## 正式入口

GPTs 创建标题以 `[literature]` 或 `[compute-literature]` 开头的 Issue，正文必须符合 `../literature-evidence-ticket.schema.json`：

```json
{
  "task_id": "literature-policy-20260729-001",
  "query": "urban congestion pricing causal effect public transport demand",
  "per_page": 10,
  "research_context": {
    "geography": "China comparable urban areas",
    "time_scope": "2015-2026",
    "outcome": "public transport demand and congestion"
  }
}
```

正式工作流 `.github/workflows/literature-evidence-ticket.yml` 绑定 GitHub Environment `compute-literature-evidence`，只执行固定 OpenAlex 检索和 Crossref DOI/更新核验。票据不能提交 URL、HTTP 头、Python、Shell、模型代码或正式数值参数。

流水线：研究问题结构化 → OpenAlex 检索 → DOI 去重 → Crossref 核验 → 更新/撤稿检查 → 可比性筛选 → 候选参数/先验 → 冻结证据包。

输出至少包括：

- `literature-evidence-package.json`；
- `literature-evidence-audit.json`；
- `literature-evidence-summary.md`；
- `artifact-manifest.json`；
- 控制台日志和结构化失败文件。

禁止把论文数值自动写入正式参数。必须区分文献原始结果、候选参数、候选先验和正式校准参数。冻结证据包只能由 GPTs 作为下一张校准票据的证据输入；文献工作流本身不能调用数值 Dispatcher。
