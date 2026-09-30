"""Synthetic metric and contrast checks."""
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from cxr_selection.evaluate import CachedMetric, contrasts


class MetricChecks(unittest.TestCase):
    def test_cache_matches_weighted_metrics_with_ties_and_zero_groups(self):
        rng=np.random.default_rng(3)
        y=rng.integers(2,size=200);score=rng.integers(10,size=200)/10
        m=CachedMetric(y,score)
        for _ in range(20):
            w=rng.integers(4,size=200)
            np.testing.assert_allclose(m(w),[roc_auc_score(y,score,sample_weight=w),average_precision_score(y,score,sample_weight=w)],atol=1e-14)
        w=np.where(score>.5,0,1)
        np.testing.assert_allclose(m(w),[roc_auc_score(y,score,sample_weight=w),average_precision_score(y,score,sample_weight=w)],atol=1e-14)

    def test_invalid_weights_and_degenerate_draw(self):
        m=CachedMetric([0,1],[.1,.9])
        with self.assertRaises(ValueError):m([-1,1])
        self.assertTrue(np.isnan(m([1,0])).all())

    def test_paired_contrast_uses_absolute_comparators(self):
        z={}
        for arm in ('explicit','random','prevalence_matched'):
            z['3072__'+arm+'__image']=np.array([.7,.3])
            z['3072__'+arm+'__fusion']=np.array([.72,.32])
        z['3072__prevalence_matched__image']=np.array([.8,.4])
        z['3072__prevalence_matched__fusion']=np.array([.81,.41])
        for arm in ('full_all','full_explicit'):
            for modality in ('image','fusion'):z['full__'+arm+'__'+modality]=np.array([.7,.3])
        c=contrasts(z)
        np.testing.assert_allclose(c['prevalence_matched_minus_explicit__fusion'],[.09,.09])
        np.testing.assert_allclose(c['prevalence_matched_minus_explicit__gain'],[-.01,-.01])
