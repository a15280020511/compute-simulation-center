#!/usr/bin/env python3
"""Bounded fixed-model data assimilation methods."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np

class StateEstimationError(ValueError):
    pass


def kalman_filter(observations: Sequence[Sequence[float]], transition, observation_matrix, process_covariance, observation_covariance, initial_state, initial_covariance) -> dict[str, Any]:
    y = np.asarray(observations, dtype=float); f = np.asarray(transition, dtype=float); h = np.asarray(observation_matrix, dtype=float)
    q = np.asarray(process_covariance, dtype=float); r = np.asarray(observation_covariance, dtype=float)
    x = np.asarray(initial_state, dtype=float); p = np.asarray(initial_covariance, dtype=float)
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[0] > 100_000 or not all(np.isfinite(v).all() for v in (y,f,h,q,r,x,p)):
        raise StateEstimationError('all inputs must be finite and observations two-dimensional')
    n = x.size
    if f.shape != (n,n) or q.shape != (n,n) or p.shape != (n,n) or h.shape[1] != n or r.shape != (h.shape[0], h.shape[0]) or y.shape[1] != h.shape[0]:
        raise StateEstimationError('matrix dimensions are inconsistent')
    states=[]; covariances=[]; innovations=[]
    eye=np.eye(n)
    for obs in y:
        x_pred=f@x; p_pred=f@p@f.T+q
        innovation=obs-h@x_pred; s=h@p_pred@h.T+r
        gain=p_pred@h.T@np.linalg.pinv(s)
        x=x_pred+gain@innovation
        p=(eye-gain@h)@p_pred@(eye-gain@h).T+gain@r@gain.T
        states.append(x.tolist()); covariances.append(p.tolist()); innovations.append(innovation.tolist())
    return {'schema_version':'kalman-filter-result-v1','method':'kalman_filter','states':states,'covariances':covariances,'innovations':innovations}


def scalar_nonlinear_filter(observations: Sequence[float], method: str, process_variance: float, observation_variance: float, initial_state: float, initial_variance: float, seed: int = 0, particles: int = 1000) -> dict[str, Any]:
    y=np.asarray(observations,dtype=float)
    if y.ndim!=1 or y.size==0 or y.size>100_000 or not np.isfinite(y).all() or process_variance<=0 or observation_variance<=0 or initial_variance<=0:
        raise StateEstimationError('invalid scalar filter inputs')
    if method not in {'extended_kalman_filter','unscented_kalman_filter','ensemble_kalman_filter','particle_filter'}:
        raise StateEstimationError('unsupported fixed scalar filter method')
    # Fixed nonlinear random walk: x_t=x_{t-1}+w, y_t=x_t^2+v.
    rng=np.random.default_rng(seed); states=[]
    if method=='particle_filter':
        count=max(100,min(int(particles),20_000)); cloud=rng.normal(initial_state, initial_variance**0.5, count)
        for obs in y:
            cloud += rng.normal(0, process_variance**0.5, count)
            logw=-0.5*((obs-cloud**2)**2/observation_variance); logw-=logw.max(); w=np.exp(logw); w/=w.sum()
            estimate=float(np.sum(w*cloud)); states.append(estimate)
            idx=rng.choice(count,size=count,p=w); cloud=cloud[idx]
    elif method=='ensemble_kalman_filter':
        count=max(100,min(int(particles),20_000)); ensemble=rng.normal(initial_state, initial_variance**0.5, count)
        for obs in y:
            ensemble += rng.normal(0, process_variance**0.5, count); predicted=ensemble**2
            cov_xy=float(np.cov(ensemble,predicted,ddof=1)[0,1]); var_y=float(np.var(predicted,ddof=1)+observation_variance)
            ensemble += cov_xy/var_y*(obs+ rng.normal(0,observation_variance**0.5,count)-predicted); states.append(float(np.mean(ensemble)))
    else:
        x=float(initial_state); p=float(initial_variance)
        for obs in y:
            p += process_variance
            if method=='extended_kalman_filter':
                predicted=x*x; jac=2*x; s=jac*jac*p+observation_variance; gain=p*jac/s; x=x+gain*(obs-predicted); p=(1-gain*jac)*p
            else:
                alpha=1e-3; lam=alpha*alpha*(1)-1; scale=max((1+lam)*p,1e-12)**0.5
                sigma=np.array([x,x+scale,x-scale]); weights=np.array([lam/(1+lam),1/(2*(1+lam)),1/(2*(1+lam))])
                z=sigma**2; zmean=float(weights@z); s=float(weights@((z-zmean)**2)+observation_variance); cov=float(weights@((sigma-x)*(z-zmean))); gain=cov/s; x=x+gain*(obs-zmean); p=max(p-gain*s*gain,1e-12)
            states.append(float(x))
    return {'schema_version':'fixed-nonlinear-filter-v1','method':method,'states':states,'fixed_transition':'random_walk','fixed_observation':'square'}
