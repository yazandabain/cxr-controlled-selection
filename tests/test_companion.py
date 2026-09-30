import unittest
import numpy as np
from cxr_selection.refit_sensitivity import shared_training_weights,patient_weights
from cxr_selection.target_transfer import masks


class CompanionTests(unittest.TestCase):
    def test_overlap_keeps_shared_multiplicity(self):
        left=np.array([1,3,5,7]); right=np.array([2,3,5,9])
        wl,wr=shared_training_weights(left,right,np.random.default_rng(31))
        self.assertEqual(wl[1],wr[1]); self.assertEqual(wl[2],wr[2])
        self.assertEqual(wl.sum()+wr.sum()-wl[1]-wl[2],6)

    def test_duplicate_training_patient_fails(self):
        with self.assertRaises(ValueError): shared_training_weights([1,1],[2,3],np.random.default_rng(2))

    def test_patient_weights_propagate_repeated_studies(self):
        w=patient_weights([1,1,1,2,3,3],np.random.default_rng(44))
        self.assertEqual(w[0],w[1]); self.assertEqual(w[1],w[2]); self.assertEqual(w[4],w[5])
        self.assertEqual(w[[0,3,4]].sum(),3)

    def test_report_mask_does_not_call_uncertain_or_missing_negative(self):
        a={part+'_report':np.array([1.,0.,-1.,np.nan]) for part in ('train','tune','test')}
        for mask in masks(a).values(): np.testing.assert_array_equal(mask,[True,True,False,False])


if __name__=='__main__': unittest.main()


