import unittest
import numpy as np
from cxr_selection.addback import sample_addback


class AddbackTests(unittest.TestCase):
    def test_unique_budget_class_counts_and_eligibility(self):
        ids=np.arange(40);states=np.repeat(['explicit_positive','explicit_negative','uncertain','blank'],10);y=np.tile([0,1],20)
        explicit=np.arange(16)
        result=sample_addback(ids,states,y,explicit,12)
        for arm,idx in result.items():
            self.assertEqual(len(idx),16);self.assertEqual(len(np.unique(ids[idx])),16)
            self.assertEqual(y[idx].sum(),8)
            excluded='blank' if arm=='explicit_uncertain' else 'uncertain'
            self.assertFalse((states[idx]==excluded).any())

    def test_patient_membership_invariant_to_input_order(self):
        ids=np.arange(40);states=np.repeat(['explicit_positive','explicit_negative','uncertain','blank'],10);y=np.tile([0,1],20)
        explicit=np.arange(16);order=np.random.default_rng(3).permutation(40)
        reordered=np.flatnonzero(np.isin(ids[order],ids[explicit]))
        left=sample_addback(ids,states,y,explicit,12);right=sample_addback(ids[order],states[order],y[order],reordered,12)
        for arm in left:np.testing.assert_array_equal(np.sort(ids[left[arm]]),np.sort(ids[order][right[arm]]))

    def test_duplicate_patients_or_invalid_anchor_fail(self):
        with self.assertRaises(ValueError):sample_addback([1,1],['explicit_positive','blank'],[1,0],[0],1)
        with self.assertRaises(ValueError):sample_addback([1,2],['explicit_positive','blank'],[1,0],[1],1)


if __name__=='__main__':unittest.main()
