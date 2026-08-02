#!/usr/bin/env python3
"""Bounded columnar data, relational join, schema and unit-analysis modes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from compute_runner import ComputeError
from think_tank_common import MAX_COLUMNS, MAX_ROWS, finite, identifiers, mapping, package, records, sequence


def bounded_table_profile(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("polars")
    package("pyarrow")
    import polars as pl

    frame = pl.DataFrame(records(inputs.get("records"), "inputs.records"), strict=False)
    if frame.width > MAX_COLUMNS:
        raise ComputeError("table contains too many columns")
    summary = []
    for column in frame.columns:
        series = frame[column]
        nulls = int(series.null_count())
        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "null_count": nulls,
            "null_rate": nulls / max(frame.height, 1),
            "unique_count": int(series.n_unique()),
        }
        if series.dtype.is_numeric():
            clean = series.drop_nulls()
            if clean.len():
                row.update(
                    mean=float(clean.mean()),
                    minimum=float(clean.min()),
                    maximum=float(clean.max()),
                    median=float(clean.median()),
                )
        summary.append(row)
    return {
        "mode": "bounded_table_profile",
        "rows": frame.height,
        "columns": frame.width,
        "summary": summary,
        "engines": {"polars": package("polars"), "pyarrow": package("pyarrow")},
        "external_data_fetches": 0,
    }


def bounded_table_join(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("duckdb")
    import duckdb
    import pandas as pd

    left = records(inputs.get("left"), "inputs.left", 20_000)
    right = records(inputs.get("right"), "inputs.right", 20_000)
    keys = identifiers(inputs.get("keys"), "inputs.keys", 10)
    how = str(inputs.get("how") or "inner")
    if how not in {"inner", "left"}:
        raise ComputeError("inputs.how must be inner or left")
    left_df = pd.DataFrame(left)
    right_df = pd.DataFrame(right)
    missing = [key for key in keys if key not in left_df.columns or key not in right_df.columns]
    if missing:
        raise ComputeError(f"join keys are missing: {missing}")
    if len(set(left_df.columns) | set(right_df.columns)) > MAX_COLUMNS:
        raise ComputeError("joined table would exceed column limit")
    conditions = " AND ".join(f'l."{key}" = r."{key}"' for key in keys)
    exclusions = ", ".join(f'"{key}"' for key in keys)
    query = f"SELECT l.*, r.* EXCLUDE ({exclusions}) FROM left_df l {how.upper()} JOIN right_df r ON {conditions}"
    try:
        result = duckdb.sql(query).df()
    except Exception as exc:
        raise ComputeError(f"bounded join failed: {type(exc).__name__}: {exc}") from exc
    if len(result) > MAX_ROWS:
        raise ComputeError("joined result exceeds row limit")
    clean = result.astype(object).where(result.notna(), None)
    return {
        "mode": "bounded_table_join",
        "how": how,
        "keys": keys,
        "row_count": int(len(result)),
        "records": clean.to_dict(orient="records"),
        "engine": {"duckdb": package("duckdb")},
        "arbitrary_sql_allowed": False,
    }


def schema_unit_validation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package("pandera")
    package("pint")
    import pandas as pd
    import pandera.pandas as pa
    from pint import UnitRegistry

    raw_schema = mapping(inputs.get("schema"), "inputs.schema")
    if len(raw_schema) > MAX_COLUMNS:
        raise ComputeError("schema contains too many columns")
    dtype_map = {"number": float, "integer": int, "string": str, "boolean": bool}
    columns: dict[str, Any] = {}
    for name, raw in raw_schema.items():
        spec = mapping(raw, f"inputs.schema[{name}]")
        dtype_name = str(spec.get("type") or "")
        if dtype_name not in dtype_map:
            raise ComputeError(f"unsupported schema type for {name}: {dtype_name}")
        checks = []
        if "minimum" in spec:
            checks.append(pa.Check.ge(finite(spec["minimum"], f"schema[{name}].minimum")))
        if "maximum" in spec:
            checks.append(pa.Check.le(finite(spec["maximum"], f"schema[{name}].maximum")))
        columns[str(name)] = pa.Column(
            dtype_map[dtype_name], checks=checks, nullable=bool(spec.get("nullable", False))
        )
    frame = pd.DataFrame(records(inputs.get("records"), "inputs.records"))
    try:
        validated = pa.DataFrameSchema(columns, strict=True).validate(frame, lazy=True)
    except Exception as exc:
        raise ComputeError(f"schema validation failed: {type(exc).__name__}: {exc}") from exc

    registry = UnitRegistry()
    conversions = []
    for i, raw in enumerate(sequence(inputs.get("unit_conversions", []), "inputs.unit_conversions")):
        spec = mapping(raw, f"inputs.unit_conversions[{i}]")
        value = finite(spec.get("value"), f"unit_conversions[{i}].value")
        source = str(spec.get("source_unit") or "")
        target = str(spec.get("target_unit") or "")
        if not source or not target:
            raise ComputeError("source_unit and target_unit are required")
        try:
            converted = (value * registry(source)).to(target)
        except Exception as exc:
            raise ComputeError(f"unit conversion failed: {source} to {target}: {exc}") from exc
        conversions.append(
            {
                "value": value,
                "source_unit": source,
                "target_unit": target,
                "converted_value": float(converted.magnitude),
            }
        )
    return {
        "mode": "schema_unit_validation",
        "validated_rows": int(len(validated)),
        "columns": list(validated.columns),
        "unit_conversions": conversions,
        "engines": {"pandera": package("pandera"), "pint": package("pint")},
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "bounded_table_profile": bounded_table_profile,
    "bounded_table_join": bounded_table_join,
    "schema_unit_validation": schema_unit_validation,
}
