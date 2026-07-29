# 专业计算能力使用指南

本文件是 GPTs 使用中心选择 `gis_spatial_analysis`、`bayesian_inference` 和 `econometric_analysis` 时的权威知识文件。

## 一、共同规则

1. 三个操作都属于计算中心，仍使用 `[compute]` Issue 和 `compute-ticket.schema.json`。
2. 计算中心不取数。坐标、GeoJSON、样本、先验、工具变量和协变量必须由用户、API中心或 GPTs 先整理好。
3. 每个变量都应在 `data_context.variables` 中记录来源、单位、观测时间、置信度和缺失情况。
4. 假设、先验、代理变量、平行趋势、排除限制等必须写入 `assumptions` 或 `limitations`，不能伪装成事实。
5. 结果只证明“给定输入和模型下的计算结果”，不自动证明现实因果关系或政策有效性。

---

## 二、专业 GIS：`gis_spatial_analysis`

### 1. 椭球测地距离矩阵

适用于经纬度点之间的地表距离，不应使用普通平面欧氏距离替代。

```json
{
  "task_id": "gis-distance-20260728",
  "operation": "gis_spatial_analysis",
  "objective": "计算候选点之间的WGS84测地距离",
  "inputs": {
    "mode": "geodesic_distance_matrix",
    "ellipsoid": "WGS84",
    "points": [
      {"id": "A", "longitude": 119.3000, "latitude": 26.0800},
      {"id": "B", "longitude": 119.3200, "latitude": 26.1000}
    ]
  }
}
```

输出单位固定为米。

### 2. 坐标系转换

```json
{
  "task_id": "gis-transform-20260728",
  "operation": "gis_spatial_analysis",
  "inputs": {
    "mode": "transform_coordinates",
    "source_crs": "EPSG:4326",
    "target_crs": "EPSG:3857",
    "points": [
      {"id": "A", "x": 119.3000, "y": 26.0800}
    ]
  }
}
```

轴顺序固定使用 `always_xy`，即经度/东坐标在前，纬度/北坐标在后。

### 3. 几何叠加

支持 `intersection`、`union`、`difference` 和 `symmetric_difference`。输入必须是 GeoJSON Geometry，不是任意文件路径或 URL。

```json
{
  "task_id": "gis-overlay-20260728",
  "operation": "gis_spatial_analysis",
  "inputs": {
    "mode": "geometry_overlay",
    "crs": "EPSG:3857",
    "action": "intersection",
    "left": {
      "type": "Polygon",
      "coordinates": [[[0,0],[100,0],[100,100],[0,100],[0,0]]]
    },
    "right": {
      "type": "Polygon",
      "coordinates": [[[50,50],[150,50],[150,150],[50,150],[50,50]]]
    }
  }
}
```

面积和长度单位取决于声明的 CRS。经纬度 CRS 中的面积或缓冲值不能直接解释为米或平方米。

### 4. 空间谓词矩阵

支持 `intersects`、`contains`、`within`、`touches`、`overlaps`、`crosses`。

### 5. 最近要素

`nearest_features` 只接受投影坐标系。若输入是 EPSG:4326，应先创建一个 `transform_coordinates` 任务转换到适合当地的投影坐标系，再计算最近距离。

### GIS 边界

当前能力是专业矢量空间运算，不包括：

- 栅格、遥感影像和地图瓦片；
- 在线地理编码、逆地理编码和实时路况；
- 大规模车辆路径规划；
- 任意 Shapefile、GeoPackage 或数据库文件上传；
- 用户自定义 Python/GDAL 命令。

---

## 三、高级贝叶斯推断：`bayesian_inference`

### 1. Beta-Binomial

适用于成功率、转化率、发生概率等二项事件。

```json
{
  "task_id": "bayes-rate-20260728",
  "operation": "bayesian_inference",
  "inputs": {
    "mode": "beta_binomial",
    "prior_alpha": 1,
    "prior_beta": 1,
    "successes": 80,
    "trials": 100,
    "credibility": 0.95
  },
  "assumptions": [
    {
      "name": "prior",
      "basis": "使用Beta(1,1)弱信息先验",
      "confidence": "medium",
      "source_type": "gpts_assumption",
      "approved_by": "user"
    }
  ]
}
```

### 2. Gamma-Poisson

适用于单位时间或单位暴露量的事件率。

### 3. 已知方差的正态均值

适用于观测噪声标准差有外部依据的连续变量均值。

### 4. 贝叶斯线性回归

使用共轭 Normal-Inverse-Gamma 模型，返回系数后验、可信区间、噪声方差后验和可选预测区间。

```json
{
  "task_id": "bayes-regression-20260728",
  "operation": "bayesian_inference",
  "inputs": {
    "mode": "bayesian_linear_regression",
    "x": [[1,10],[2,12],[3,15],[4,17]],
    "y": [20,24,31,35],
    "add_intercept": true,
    "coefficient_names": ["intercept","price","traffic"],
    "prior_precision": 0.001,
    "prior_shape": 0.001,
    "prior_scale": 0.001,
    "credibility": 0.95,
    "x_new": [[5,20]]
  }
}
```

### 贝叶斯边界

当前设计优先可审计、可重复和低维护，因此不开放：

- 用户自定义概率程序；
- 任意似然函数；
- 无边界 MCMC；
- PyMC/Stan 模型代码执行；
- 运行时安装插件。

需要复杂层级模型、空间贝叶斯模型或非共轭后验时，应先形成固定需求，再新增独立白名单操作和验收基准，不能通过票据注入代码。

---

## 四、专业计量经济学：`econometric_analysis`

### 1. OLS / WLS

支持 `NONROBUST`、`HC0`、`HC1`、`HC2`、`HC3` 协方差。默认 `HC1`。

```json
{
  "task_id": "econ-ols-20260728",
  "operation": "econometric_analysis",
  "inputs": {
    "mode": "ols",
    "x": [[1,10],[2,12],[3,15],[4,17],[5,20]],
    "y": [20,24,31,35,43],
    "add_intercept": true,
    "coefficient_names": ["intercept","price","traffic"],
    "covariance_type": "HC1",
    "credibility": 0.95
  }
}
```

### 2. 双重差分

```json
{
  "task_id": "econ-did-20260728",
  "operation": "econometric_analysis",
  "inputs": {
    "mode": "difference_in_differences",
    "outcome": [10,12,11,17,9,10,10,16],
    "treatment": [0,0,1,1,0,0,1,1],
    "post": [0,1,0,1,0,1,0,1],
    "covariance_type": "HC1"
  },
  "limitations": [
    "因果解释依赖平行趋势、无差异性同期冲击和稳定处理定义。"
  ]
}
```

输出中的 `treatment_x_post` 是双重差分估计量，但计算中心不会宣称平行趋势已经成立。

### 3. 工具变量 2SLS

输入拆分为：

- `exogenous`：外生控制变量；
- `endogenous`：内生解释变量；
- `instruments`：排除工具变量；
- `y`：被解释变量。

```json
{
  "task_id": "econ-iv-20260728",
  "operation": "econometric_analysis",
  "inputs": {
    "mode": "iv_2sls",
    "y": [10,12,15,18,21,24],
    "exogenous": [[1],[2],[3],[4],[5],[6]],
    "endogenous": [[2],[3],[5],[6],[8],[9]],
    "instruments": [[1],[1.5],[2.5],[3],[4],[5]],
    "covariance_type": "HC1"
  },
  "assumptions": [
    {
      "name": "instrument_exclusion",
      "basis": "工具变量只通过内生解释变量影响结果",
      "confidence": "medium",
      "source_type": "expert_hypothesis",
      "approved_by": "user"
    }
  ]
}
```

第一阶段 F 统计量只检查相关性强弱，不证明工具变量满足排除限制或外生性。

### 计量边界

当前不包括：

- 自动选择“最有利”的模型规格；
- 根据结果反复试验变量直到显著；
- 未声明的缺失值填补；
- 自动因果结论；
- 大型面板、多层模型、空间计量和结构估计；
- 任意公式字符串或用户代码执行。

---

## 五、与专家团和 API 中心组合

三个中心是并列模块，不直接互调。常见组合由 GPTs 完成：

```text
API中心取得结构化公开数据
→ GPTs核验正文、Artifact、Manifest和SHA
→ 创建新的[compute]票据
→ GPTs核验计算结果
→ 创建新的[execution]票据交给专家团解释和裁决
```

也可以：

- 只有数据计算：只用计算中心；
- 先由专家提出待检验假设，再由 GPTs 整理成新计算票据；
- GIS 和计量任务彼此独立时并行执行；
- 存在数据依赖时必须串行。

箭头表示 GPTs 搬运、核验和重新出票，不表示中心之间建立了直接调用链。
