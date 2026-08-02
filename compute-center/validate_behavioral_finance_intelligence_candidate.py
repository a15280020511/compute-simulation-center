#!/usr/bin/env python3
"""Isolated compatibility probes for the final behavioral/finance/intelligence gaps.

This temporary validator never fetches data, accepts no ticket input, and does not
exercise agentic, MCP, brokerage, remote-model, or arbitrary-code features.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import scipy

EXPECTED = {
    "quantlib": ("QuantLib", "1.43"),
    "egttools": ("egttools", "0.1.14.2"),
    "pymdp": ("inferactively-pymdp", "1.0.3"),
    "psyneulink": ("psyneulink", "0.18.0.0"),
    "pyod": ("pyod", "3.6.2"),
    "graspologic": ("graspologic", "3.4.4"),
    "mlxtend": ("mlxtend", "0.25.0"),
}


def quantlib_probe() -> dict:
    import QuantLib as ql

    today = ql.Date(3, ql.August, 2026)
    ql.Settings.instance().evaluationDate = today
    maturity = ql.Date(3, ql.August, 2027)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    rate = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, ql.Actual365Fixed()))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot, dividend, rate, volatility)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    npv = float(option.NPV())
    delta = float(option.delta())
    assert 0.0 < npv < 100.0 and 0.0 < delta < 1.0
    return {"european_call_npv": npv, "delta": delta}


def egttools_probe() -> dict:
    import egttools

    public = sorted(name for name in dir(egttools) if not name.startswith("_"))
    assert public
    return {"public_symbol_count": len(public), "sample_symbols": public[:10]}


def pymdp_probe() -> dict:
    import pymdp
    from pymdp.agent import Agent

    assert Agent is not None
    public = sorted(name for name in dir(pymdp) if not name.startswith("_"))
    return {
        "agent_class_available": True,
        "public_symbol_count": len(public),
        "autonomous_loop_executed": False,
    }


def psyneulink_probe() -> dict:
    import psyneulink as pnl

    mechanism = pnl.TransferMechanism(function=pnl.Linear(slope=2.0, intercept=1.0))
    value = np.asarray(mechanism.execute([3.0]), dtype=float).reshape(-1)
    assert value.size == 1 and abs(float(value[0]) - 7.0) < 1e-8
    return {"transfer_output": float(value[0]), "custom_model_code_used": False}


def pyod_probe() -> dict:
    from pyod.models.iforest import IForest

    observations = np.asarray(
        [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.05, 0.0], [8.0, 8.0]],
        dtype=float,
    )
    detector = IForest(n_estimators=50, contamination=0.2, random_state=7)
    detector.fit(observations)
    scores = np.asarray(detector.decision_scores_, dtype=float)
    assert scores.shape == (5,) and int(np.argmax(scores)) == 4
    return {
        "detector": "IForest",
        "highest_anomaly_index": int(np.argmax(scores)),
        "agentic_engine_used": False,
        "mcp_used": False,
    }


def graspologic_probe() -> dict:
    from graspologic.embed import AdjacencySpectralEmbed

    adjacency = np.asarray(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=float,
    )
    latent = AdjacencySpectralEmbed(n_components=2).fit_transform(adjacency)
    if isinstance(latent, tuple):
        latent = latent[0]
    latent = np.asarray(latent, dtype=float)
    assert latent.shape == (6, 2) and np.isfinite(latent).all()
    return {"embedding_shape": list(latent.shape), "finite": True}


def mlxtend_probe() -> dict:
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules

    frame = pd.DataFrame(
        [
            {"bread": True, "milk": True, "eggs": False},
            {"bread": True, "milk": True, "eggs": True},
            {"bread": True, "milk": False, "eggs": True},
            {"bread": False, "milk": True, "eggs": True},
        ]
    )
    frequent = apriori(frame, min_support=0.5, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)
    assert len(frequent) >= 3 and not rules.empty
    return {"frequent_itemsets": int(len(frequent)), "association_rules": int(len(rules))}


PROBES = {
    "quantlib": quantlib_probe,
    "egttools": egttools_probe,
    "pymdp": pymdp_probe,
    "psyneulink": psyneulink_probe,
    "pyod": pyod_probe,
    "graspologic": graspologic_probe,
    "mlxtend": mlxtend_probe,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(PROBES), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    distribution, expected_version = EXPECTED[args.candidate]
    observed_version = importlib.metadata.version(distribution)
    assert observed_version == expected_version, (observed_version, expected_version)
    assert np.__version__ == "2.4.6", np.__version__
    assert scipy.__version__ == "1.18.0", scipy.__version__

    result = {
        "status": "PASS",
        "candidate": args.candidate,
        "distribution": distribution,
        "version": observed_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "functional_result": PROBES[args.candidate](),
        "network_used": False,
        "model_calls": 0,
        "arbitrary_code_used": False,
        "remote_data_used": False,
        "brokerage_execution_used": False,
        "personal_psychological_diagnosis_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
