import unittest
import numpy as np
from cxr_selection.group_contrast import paired_interval

class PairedContrastTests(unittest.TestCase):
    def test_constant_paired_difference(self):
        x=np.arange(100,dtype=float)/100
        np.testing.assert_allclose(paired_interval(x+.2,x),[.2,.2],atol=1e-14)
    def test_reversed_shared_draws(self):
        x=np.arange(100,dtype=float)/100
        np.testing.assert_allclose(paired_interval(x,-x),np.quantile(2*x,[.025,.975]))
    def test_rejects_bad_alignment(self):
        with self.assertRaises(ValueError):paired_interval([1,2],[1,2,3])
        with self.assertRaises(ValueError):paired_interval([1,np.nan],[1,2])
if __name__=='__main__':unittest.main()


