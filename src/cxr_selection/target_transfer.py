"""Explicit report-state eligibility masks."""
import numpy as np


def masks(a):
    return {part: np.isin(a[part+'_report'], [0, 1]) for part in ('train','tune','test')}
