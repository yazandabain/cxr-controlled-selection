"""Percentile intervals from paired bootstrap contrasts."""
import numpy as np


def paired_interval(other,explicit):
    other,explicit=np.asarray(other),np.asarray(explicit)
    if other.ndim!=1 or other.shape!=explicit.shape or len(other)<2:
        raise ValueError('Expected aligned nonempty one-dimensional paired draws')
    if not np.isfinite(other).all() or not np.isfinite(explicit).all():raise ValueError('Nonfinite draws')
    return np.quantile(other-explicit,[.025,.975]).tolist()
