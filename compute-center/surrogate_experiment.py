#!/usr/bin/env python3
"""Bounded experiment-design and surrogate validation utilities."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np

class SurrogateError(ValueError): pass


def latin_hypercube_design(bounds: Sequence[Sequence[float]], samples: int, seed: int = 0) -> dict[str, Any]:
    b=np.asarray(bounds,dtype=float)
    if b.ndim!=2 or b.shape[1]!=2 or b.shape[0]==0 or b.shape[0]>50 or samples<2 or samples>100_000 or not np.isfinite(b).all() or np.any(b[:,0]>=b[:,1]):
        raise SurrogateError('invalid bounds or sample count')
    rng=np.random.default_rng(seed); n=int(samples); d=b.shape[0]; design=np.empty((n,d))
    for j in range(d):
        points=(np.arange(n)+rng.random(n))/n; rng.shuffle(points); design[:,j]=b[j,0]+points*(b[j,1]-b[j,0])
    return {'schema_version':'latin-hypercube-v1','seed':seed,'design':design.tolist()}


def gaussian_process_surrogate(x, y, x_predict, seed: int = 0) -> dict[str, Any]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    xx=np.asarray(x,dtype=float); yy=np.asarray(y,dtype=float); xp=np.asarray(x_predict,dtype=float)
    if xx.ndim!=2 or yy.ndim!=1 or xx.shape[0]!=yy.size or xp.ndim!=2 or xp.shape[1]!=xx.shape[1] or xx.shape[0]>20_000:
        raise SurrogateError('invalid surrogate arrays')
    model=GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(nu=2.5)+WhiteKernel(1e-6),normalize_y=True,random_state=seed,n_restarts_optimizer=1)
    model.fit(xx,yy); mean,std=model.predict(xp,return_std=True)
    return {'schema_version':'gaussian-process-surrogate-v1','prediction':mean.tolist(),'standard_deviation':std.tolist(),'kernel':str(model.kernel_)}


def polynomial_chaos_surrogate(x, y, x_predict, degree: int = 2) -> dict[str, Any]:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    xx=np.asarray(x,dtype=float); yy=np.asarray(y,dtype=float); xp=np.asarray(x_predict,dtype=float)
    if degree<1 or degree>4 or xx.ndim!=2 or yy.ndim!=1 or xx.shape[0]!=yy.size or xp.ndim!=2 or xp.shape[1]!=xx.shape[1]: raise SurrogateError('invalid polynomial surrogate inputs')
    poly=PolynomialFeatures(degree=degree,include_bias=True); tx=poly.fit_transform(xx); model=LinearRegression().fit(tx,yy); pred=model.predict(poly.transform(xp))
    return {'schema_version':'polynomial-surrogate-v1','degree':degree,'prediction':pred.tolist(),'training_r2':float(model.score(tx,yy))}


def surrogate_error_validation(observed, predicted, maximum_relative_rmse: float = .1) -> dict[str, Any]:
    y=np.asarray(observed,dtype=float); p=np.asarray(predicted,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or y.size==0 or not np.isfinite(y).all() or not np.isfinite(p).all(): raise SurrogateError('observed and predicted must be equal finite vectors')
    rmse=float(np.sqrt(np.mean((y-p)**2))); scale=float(np.std(y)) or max(float(np.mean(np.abs(y))),1e-12); relative=rmse/scale
    return {'schema_version':'surrogate-validation-v1','rmse':rmse,'relative_rmse':relative,'publish_allowed':relative<=maximum_relative_rmse,'real_model_recheck_required':True}


def active_learning(candidates, mean, standard_deviation, batch_size: int = 1) -> dict[str, Any]:
    x=np.asarray(candidates,dtype=float); mu=np.asarray(mean,dtype=float); sd=np.asarray(standard_deviation,dtype=float)
    if x.ndim!=2 or mu.shape!=(x.shape[0],) or sd.shape!=(x.shape[0],) or batch_size<1 or batch_size>x.shape[0]: raise SurrogateError('invalid active-learning inputs')
    idx=np.argsort(-sd)[:batch_size]
    return {'schema_version':'active-learning-v1','criterion':'maximum-predictive-standard-deviation','selected_indices':idx.tolist(),'selected_candidates':x[idx].tolist(),'selected_uncertainty':sd[idx].tolist()}
