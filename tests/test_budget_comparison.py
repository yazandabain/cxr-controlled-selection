import unittest
import numpy as np
from cxr_selection.budget_comparison import simultaneous_band


class BudgetTests(unittest.TestCase):
    def test_family_band_covers_bootstrap_deviations(self):
        rng=np.random.default_rng(123)
        point=np.array([.1,-.2,.3]);draws=point+rng.normal(size=(1000,3))*.02
        band,q=simultaneous_band(point,draws)
        self.assertGreater(q,1.96)
        coverage=((draws>=band[:,0])&(draws<=band[:,1])).all(axis=1).mean()
        self.assertAlmostEqual(coverage,.95,places=2)

    def test_identical_contrasts_do_not_inflate_family(self):
        x=np.random.default_rng(123).normal(size=1000)
        one,q1=simultaneous_band(np.zeros(1),x[:,None])
        many,q2=simultaneous_band(np.zeros(12),np.repeat(x[:,None],12,axis=1))
        self.assertAlmostEqual(q1,q2)
        np.testing.assert_allclose(many,np.repeat(one,12,axis=0))

    def test_invalid_or_degenerate_inputs_fail(self):
        with self.assertRaises(ValueError):simultaneous_band([0,1],np.zeros((5,1)))
        with self.assertRaises(ValueError):simultaneous_band([0],np.ones((5,1)))


if __name__=='__main__':unittest.main()
