#!/usr/bin/env python3
"""Append-only prediction/realization matcher and calibration/drift evaluator."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

class FeedbackError(ValueError): pass


def evaluate(records: list[dict[str, Any]], drift_window: int = 20) -> dict[str, Any]:
    matched=[r for r in records if r.get('record_type') in {'realized_outcome','shadow_result'} and 'prediction' in r and 'realized' in r]
    if not matched: raise FeedbackError('no matched prediction/realized records')
    errors=[]; squared=[]; brier=[]; covered=[]
    for row in matched:
        prediction=float(row['prediction']); realized=float(row['realized'])
        if not math.isfinite(prediction) or not math.isfinite(realized): raise FeedbackError('non-finite record')
        error=prediction-realized; errors.append(error); squared.append(error*error)
        if row.get('kind')=='probability':
            if not 0<=prediction<=1 or realized not in {0,1}: raise FeedbackError('invalid probability record')
            brier.append(error*error)
        if row.get('interval_lower') is not None and row.get('interval_upper') is not None:
            lo=float(row['interval_lower']); hi=float(row['interval_upper']); covered.append(lo<=realized<=hi)
    recent=np.asarray(errors[-drift_window:],dtype=float); earlier=np.asarray(errors[:-drift_window],dtype=float)
    drift=None if earlier.size<drift_window or recent.size<drift_window else abs(float(np.mean(recent)-np.mean(earlier))) / max(float(np.std(earlier)),1e-12)
    return {'schema_version':'feedback-evaluation-v1','matched_records':len(matched),'mae':float(np.mean(np.abs(errors))),'rmse':float(np.sqrt(np.mean(squared))),'mean_error':float(np.mean(errors)),'brier':None if not brier else float(np.mean(brier)),'interval_coverage':None if not covered else float(np.mean(covered)),'standardized_mean_drift':drift,'action':'REVALIDATE' if drift is not None and drift>1.0 else 'CONTINUE'}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--ledger',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.ledger).read_text(encoding='utf-8').splitlines() if line.strip()]
    Path(a.output).write_text(json.dumps(evaluate(rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
