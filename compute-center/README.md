# Independent Compute Center

This directory is a self-contained deterministic computation module controlled by GPTs.

## Architectural boundary

- GPTs collects public information, records user requirements, and explicitly writes assumptions.
- The compute center validates one fixed JSON operation, runs deterministic data preflight, and executes only the selected allowlisted algorithm.
- The API center and Open Model Market remain separate peer modules with their own Issue prefixes, workflows, dependencies, logs, state, and Artifacts.
- The centers never call each other. GPTs is the only cross-center relay.
- GPTs may transfer the complete `compute-result.json` package to the expert team as evidence.
- No model, search service, external API, arbitrary Python, shell command, URL fetch, or runtime plugin installation is executed by this module.

## Entry point

Create a repository Issue whose title starts with:

```text
[compute]
```

The Issue body must conform to `compute-ticket.schema.json`.

## Data preflight

Every accepted compute task first creates `compute-preflight.json`. The preflight checks:

- missing required variables and replacement strategies;
- observed, user, public, historical, benchmark, proxy and assumption sources;
- units, timestamps, sample sizes, missing counts and confidence;
- low-confidence assumptions, ranges and user approval;
- probability ranges and sums, non-finite values and division-by-zero risks;
- recommended representative values such as mean, median, trimmed mean, weighted mean, grouped summaries or intervals;
- recommended follow-up operations such as sensitivity, scenario or Monte Carlo analysis.

Preflight states:

- `DATA_READY`
- `DATA_READY_WITH_ASSUMPTIONS`
- `DATA_DEGRADED`
- `USER_APPROVAL_REQUIRED`
- `DATA_INSUFFICIENT`

`USER_APPROVAL_REQUIRED` and `DATA_INSUFFICIENT` block the operation. GPTs must resolve the data gap or create a new approved ticket. The compute center never fetches missing data itself.

Authoritative policy files:

- `data-readiness-policy.json`
- `data-readiness-playbook.md`
- `compute-preflight.schema.json`
- `compute-capabilities.json`

## Supported operations

### Core decision and simulation operations

- `monte_carlo`
- `sensitivity_analysis`
- `scenario_compare`
- `constrained_optimization`
- `break_even_analysis`
- `descriptive_statistics`
- `discrete_event_simulation`
- `repeated_game`
- `agent_evolution`
- `time_series_forecast`
- `causal_screening`
- `nonlinear_dynamics`
- `pattern_discovery`
- `assumption_validation`
- `markov_simulation`

### Professional GIS

`gis_spatial_analysis` supports fixed modes:

- `geodesic_distance_matrix`: WGS84 or another declared ellipsoid;
- `transform_coordinates`: CRS-aware coordinate transformation with fixed axis order;
- `geometry_overlay`: intersection, union, difference and symmetric difference for GeoJSON geometries;
- `spatial_predicate_matrix`: intersects, contains, within, touches, overlaps and crosses;
- `nearest_features`: bounded nearest-feature calculation in a projected CRS.

The GIS operation is vector-only. Raster processing, live routing, map tile services, arbitrary file ingestion and online geocoding remain outside the compute center.

### Advanced Bayesian inference

`bayesian_inference` supports fixed auditable models:

- `beta_binomial`;
- `gamma_poisson`;
- `normal_mean_known_variance`;
- `bayesian_linear_regression` using a conjugate Normal-Inverse-Gamma formulation.

The operation returns posterior parameters, credible intervals and optional predictive intervals. It does not run arbitrary probabilistic programs, user-supplied likelihood code or unrestricted MCMC.

### Professional econometrics

`econometric_analysis` supports:

- `ols`;
- `wls`;
- `difference_in_differences`;
- `iv_2sls`.

Available diagnostics include robust covariance choices, confidence intervals, residual diagnostics, condition number, first-stage excluded-instrument statistics and explicit identification warnings. Statistical output does not by itself establish causal validity.

The GPTs-facing selection catalog is `compute-capabilities.json`. Complete usage templates are in `professional-operations-guide.md`.

## Example

```json
{
  "task_id": "compute-example-20260728",
  "objective": "Estimate the break-even volume.",
  "operation": "break_even_analysis",
  "inputs": {
    "fixed_cost": 100000,
    "unit_price": 80,
    "variable_cost": 50
  },
  "data_context": {
    "variables": [
      {
        "name": "unit_price",
        "required": true,
        "source_type": "gpts_assumption",
        "unit": "CNY/unit",
        "confidence": "medium",
        "missing": false,
        "replacement_strategy": "none"
      }
    ]
  },
  "assumptions": [
    {
      "name": "unit_price",
      "value": 80,
      "unit": "CNY/unit",
      "basis": "User-provided planning assumption",
      "confidence": "medium",
      "source_type": "user_assumption",
      "sensitivity_range": {"minimum": 70, "maximum": 90},
      "invalid_when": "actual transaction prices become available",
      "approved_by": "user"
    }
  ]
}
```

## Output and diagnostic contract

The workflow publishes:

- `compute-preflight.json`: data-readiness status, gaps, representative-value recommendations and suggested operations.
- `compute-result.json`: complete transfer package for GPTs or another independent consumer.
- `compute-audit.json`: input/output hashes, timing, and zero-model/zero-network declarations.
- `compute-diagnostics.json`: stage matrix, run identity, primary failure, runtime versions, evidence inventory, and remediation hints.
- `compute-summary.md`: compact human-readable status.
- `compute-error.json`: structured failure evidence with error code, stage, traceback, task/operation correlation, retryability, and safe runtime context.
- `compute-console.log`: complete dispatcher console output captured by the workflow.
- `artifact-manifest.json`: SHA-256 inventory of the Artifact.

`compute-result.json` is the supported numerical data-transfer contract. GPTs should preserve it verbatim when forwarding results to the expert team and should include `compute-preflight.json` so experts can see data quality and assumptions.

## Limits and toolkit sufficiency

- Up to 100,000 Monte Carlo iterations.
- Up to 50 variables.
- Up to 50 scenarios.
- Up to 200 optimization constraints.
- Up to 100,000 descriptive-statistics values.
- Up to 1,000 GIS points or 500 geometries per side, with at most 100,000 pairwise comparisons.
- Up to 100,000 Bayesian observations.
- Up to 100,000 econometric rows and 100 submitted columns.
- No arbitrary formulas or code execution.
- No external data retrieval.

The toolkit intentionally uses `jsonschema + NumPy + SciPy + SimPy + Shapely + PyProj`. Bayesian and econometric operations reuse NumPy and SciPy to limit dependency growth. GIS adds only vector geometry and CRS packages, not GeoPandas, GDAL, raster stacks or routing engines.

## Maintenance

Dependencies are isolated in `compute-center/requirements.txt`. Dependabot checks this directory separately. Compute validation is path-scoped and does not run the Open Model Market. Ordinary Web GPT with the GitHub plugin is the maintenance role; GPTs is the usage role.

Recovery, rebuild and external-configuration backup instructions are maintained under `recovery/`.

## 金融与商业决策能力

`finance_decision_analysis`提供收益风险指标、投资组合优化、复利投资投影、商业单元经济和资本预算。它只处理票据中已提供的数据，不取实时行情、不连接券商、不执行交易，也不承诺收益。PyPortfolioOpt仅在选择该操作时安装。

## 工具注册与金融扩展

- `tool-registry.json` 是计算中心唯一的扩展工具登记表；`tool_registry.py` 负责固定模块注册、冲突检查和按票据选择依赖。
- 新工具必须是仓库内固定模块、固定操作、固定依赖文件，不接受票据提供模块名、包名、Python代码或运行时安装命令。
- `finance_decision_analysis` 已覆盖风险收益、组合优化、投资测算、商业单位经济、资本预算和固定策略回测。
- 固定策略回测使用隔离的 `vectorbt==1.1.0`，只开放买入持有与均线交叉，不联网、不画图、不执行任意策略代码。

## Global Think-Tank Toolkit

Exa and Tavily global discovery produced 16 candidates. Fifteen passed isolated Python 3.12 compatibility and numerical smoke tests; CausalPy was excluded because it requires NumPy below the production baseline. The accepted modes remain fixed, bounded, repository-pinned, network-denied and controlled-preview.
