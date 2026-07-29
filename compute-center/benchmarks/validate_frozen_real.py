#!/usr/bin/env python3
"""Validate real-world frozen reference datasets and decision-quality behavior."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_operations import time_series_forecast  # noqa: E402
from professional_operations import econometric_analysis  # noqa: E402
from quality_gate import evaluate_feedback  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "frozen-real" / "manifest.json"


def dataframe_sha256(frame: object) -> str:
    records = frame.to_dict(orient="records")
    raw = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sm.__version__ != manifest["statsmodels_version"]:
        raise AssertionError(f"statsmodels version drift: {sm.__version__}")

    loaders = {
        "longley-us-macro-1947-1962": sm.datasets.longley,
        "nile-flow-1871-1970": sm.datasets.nile,
        "sunspots-1700-2008": sm.datasets.sunspots,
        "us-macro-quarterly-1959-2009": sm.datasets.macrodata,
        "spector-education-binary-outcomes": sm.datasets.spector,
    }
    frames = {}
    rows: list[dict[str, object]] = []
    for spec in manifest["datasets"]:
        frame = loaders[spec["id"]].load_pandas().data
        observed = dataframe_sha256(frame)
        if observed != spec["sha256"]:
            raise AssertionError(f"dataset hash mismatch for {spec['id']}: {observed}")
        frames[spec["id"]] = frame
        rows.append({"id": spec["id"], "hash_status": "PASS", "rows": int(frame.shape[0]), "columns": int(frame.shape[1])})

    longley = frames["longley-us-macro-1947-1962"]
    longley_result = econometric_analysis({
        "mode": "ols",
        "x": longley[["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]].values.tolist(),
        "y": longley["TOTEMP"].values.tolist(),
        "covariance_type": "HC1",
    })
    if float(longley_result["r_squared"]) < float(manifest["acceptance"]["longley_r_squared_minimum"]):
        raise AssertionError("Longley regression R-squared below benchmark threshold")
    if float(longley_result.get("condition_number", 0.0)) < 1e6:
        raise AssertionError("Longley benchmark must expose severe ill-conditioning")

    nile = frames["nile-flow-1871-1970"]
    nile_result = time_series_forecast({"data": nile["volume"].values.tolist(), "horizon": 5, "holdout": 20})
    if float(nile_result["selected_method"]["mae"]) > float(manifest["acceptance"]["nile_holdout_mae_maximum"]):
        raise AssertionError("Nile holdout MAE exceeds benchmark threshold")
    if len(nile_result["forecast"]) != 5:
        raise AssertionError("Nile forecast horizon mismatch")

    sunspots = frames["sunspots-1700-2008"]["SUNACTIVITY"].to_numpy(dtype=float)
    lag11 = float(np.corrcoef(sunspots[:-11], sunspots[11:])[0, 1])
    if lag11 < float(manifest["acceptance"]["sunspots_lag11_correlation_minimum"]):
        raise AssertionError("Sunspot cycle benchmark was not recovered")

    spector = frames["spector-education-binary-outcomes"]
    design = sm.add_constant(spector[["GPA", "TUCE", "PSI"]])
    fit = sm.Logit(spector["GRADE"], design).fit(disp=False)
    probabilities = fit.predict(design).tolist()
    feedback = evaluate_feedback({
        "calibration_feedback": {
            "predicted_probabilities": probabilities,
            "observed_outcomes": spector["GRADE"].astype(int).tolist(),
            "feedback_window": "Spector frozen dataset",
        }
    })
    brier = float(feedback["probability_calibration"]["brier_score"])
    if brier > float(manifest["acceptance"]["spector_brier_score_maximum"]):
        raise AssertionError("Spector probability calibration benchmark failed")

    macro = frames["us-macro-quarterly-1959-2009"]["realgdp"].to_numpy(dtype=float)
    drift = evaluate_feedback({
        "calibration_feedback": {
            "reference_values": macro[:80].tolist(),
            "recent_values": macro[-80:].tolist(),
            "feedback_window": "Early versus late US macro real GDP",
        }
    })["drift"]
    severe_drift = (
        float(drift["population_stability_index"]) > 0.25
        or float(drift["kolmogorov_smirnov_statistic"]) > 0.3
        or float(drift["standardized_mean_shift"]) > 1.0
    )
    if bool(manifest["acceptance"]["macrodata_drift_must_block"]) and not severe_drift:
        raise AssertionError("Macrodata benchmark failed to trigger severe drift")

    return {
        "schema_version": "frozen-real-benchmark-result-v1",
        "status": "PASS",
        "datasets": rows,
        "metrics": {
            "longley_r_squared": longley_result["r_squared"],
            "longley_condition_number": longley_result.get("condition_number"),
            "nile_selected_method": nile_result["selected_method"],
            "sunspots_lag11_correlation": lag11,
            "spector_brier_score": brier,
            "macrodata_drift": drift,
        },
    }


if __name__ == "__main__":
    output = run()
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frozen-real-benchmark-result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
