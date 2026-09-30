import unittest
import numpy as np
from cxr_selection.pair_accounting import PairAccounting
from cxr_selection.models import weighted_auc_ap


def brute(y,score,state,w,groups):
    numerator=np.zeros((groups,groups)); denominator=np.zeros_like(numerator)
    for i in np.flatnonzero(y==1):
        for j in np.flatnonzero(y==0):
            mass=w[i]*w[j]
            denominator[state[i],state[j]]+=mass
            numerator[state[i],state[j]]+=mass*((score[i]>score[j])+.5*(score[i]==score[j]))
    return numerator/denominator.sum(),denominator/denominator.sum()


class PairTests(unittest.TestCase):
    def test_ties_and_weighted_pairs(self):
        rng=np.random.default_rng(911)
        y=np.tile([0,1],30);score=rng.integers(0,5,60)/4;state=rng.integers(0,4,60)
        metric=PairAccounting(y,score,state)
        for _ in range(20):
            w=rng.integers(0,4,60)
            result=metric(w);numerator,mass=brute(y,score,state,w,4)
            np.testing.assert_allclose(result['contribution'],numerator,atol=1e-14)
            np.testing.assert_allclose(result['weight'],mass,atol=1e-14)
            self.assertAlmostEqual(result['contribution'].sum(),weighted_auc_ap(y,score,w)['auroc'],places=14)

    def test_empty_cell_is_unavailable_not_zero_auc(self):
        metric=PairAccounting(np.array([1,0]),np.array([.8,.2]),np.array([0,1]))
        result=metric(np.ones(2))
        self.assertTrue(np.isnan(result['cell_auc'][2,2]))
        self.assertEqual(result['contribution'][2,2],0)
        self.assertEqual(result['cell_auc'][0,1],1)

    def test_shared_patient_multiplicity_and_contrast_identity(self):
        y=np.array([1,0,1,0,1,0]);g=np.array([0,1,1,2,2,3]);patient=np.array([0,0,1,1,2,3])
        a=np.array([.8,.4,.6,.7,.2,.2]);b=np.array([.9,.2,.6,.8,.5,.2])
        w=np.array([2,0,1,3])[patient]
        left,right=PairAccounting(y,a,g)(w),PairAccounting(y,b,g)(w)
        diff=right['contribution']-left['contribution']
        expected=weighted_auc_ap(y,b,w)['auroc']-weighted_auc_ap(y,a,w)['auroc']
        self.assertAlmostEqual(diff.sum(),expected,places=14)
        self.assertAlmostEqual(np.trace(diff)+(diff.sum()-np.trace(diff)),expected,places=14)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PairAccounting(np.array([1,2]),np.array([.1,.2]),np.array([0,1]))
        metric=PairAccounting(np.array([1,0]),np.array([.1,.2]),np.array([0,1]))
        with self.assertRaises(ValueError):metric([-1,1])
        with self.assertRaises(ValueError):metric([0,1])


if __name__=='__main__':unittest.main()
