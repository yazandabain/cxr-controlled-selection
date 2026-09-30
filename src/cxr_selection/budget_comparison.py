"""Approximate simultaneous bands from aligned bootstrap draws."""
import numpy as np


def simultaneous_band(point, draws, coverage=.95):
    point,draws=np.asarray(point,dtype=float),np.asarray(draws,dtype=float)
    if point.ndim!=1 or draws.ndim!=2 or draws.shape[1]!=len(point) or len(draws)<2:
        raise ValueError('Unexpected bootstrap dimensions')
    if not np.isfinite(draws).all() or not np.isfinite(point).all():
        raise ValueError('Nonfinite bootstrap statistic')
    sd=draws.std(axis=0,ddof=1)
    z=np.divide(np.abs(draws-point),sd,out=np.zeros_like(draws),where=sd>0)
    if ((sd==0)&np.any(draws!=point,axis=0)).any():
        raise ValueError('Degenerate bootstrap cannot estimate centered uncertainty')
    critical=float(np.quantile(z.max(axis=1),coverage))
    return np.column_stack((point-critical*sd,point+critical*sd)),critical
