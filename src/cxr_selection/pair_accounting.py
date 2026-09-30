"""Exact tie-aware AUROC accounting by group pairs."""
import numpy as np


class PairAccounting:
    """Tie-aware weighted pair numerators; no patient arrays are written here."""
    def __init__(self, y, score, state, groups=4):
        y, score, state = np.asarray(y), np.asarray(score), np.asarray(state)
        if y.ndim != 1 or score.shape != y.shape or state.shape != y.shape:
            raise ValueError('Mismatched input shapes')
        if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
            raise ValueError('Invalid label/score')
        if not np.issubdtype(state.dtype, np.integer) or (state < 0).any() or (state >= groups).any():
            raise ValueError('Invalid state codes')
        self.order = np.argsort(-score, kind='stable')
        bins = np.r_[0, np.cumsum(np.diff(score[self.order]) != 0)]
        self.rows, self.groups = int(bins[-1]) + 1, groups
        self.pos = y[self.order] == 1
        index = bins * groups + state[self.order]
        self.positive_index, self.negative_index = index[self.pos], index[~self.pos]

    def __call__(self, weights):
        w = np.asarray(weights, dtype=float)
        if w.shape != self.order.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('Invalid weights')
        w = w[self.order]
        positive = np.bincount(self.positive_index, weights=w[self.pos], minlength=self.rows*self.groups).reshape(self.rows,self.groups)
        negative = np.bincount(self.negative_index, weights=w[~self.pos], minlength=self.rows*self.groups).reshape(self.rows,self.groups)
        positive_total, negative_total = positive.sum(axis=0), negative.sum(axis=0)
        denominator = np.outer(positive_total, negative_total)
        total = denominator.sum()
        if total <= 0:
            raise ValueError('Both coded classes are required')
        numerators = positive.T @ (negative_total - negative.cumsum(axis=0) + .5*negative)
        cell_auc = np.divide(numerators, denominator, out=np.full_like(numerators, np.nan), where=denominator > 0)
        return {'contribution': numerators/total, 'weight': denominator/total, 'cell_auc': cell_auc}
