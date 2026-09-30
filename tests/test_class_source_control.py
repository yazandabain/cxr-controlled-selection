import unittest
import numpy as np
from cxr_selection.class_source_control import cross_memberships, factorial_contrasts


class ClassSourceTests(unittest.TestCase):
    def test_exact_class_sources_and_no_duplicates(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        e = np.array([0, 1, 2, 3]); b = np.array([2, 3, 4, 5])
        crossed = cross_memberships(y, e, b)
        np.testing.assert_array_equal(crossed['p1n0'], [2, 4, 1, 3])
        np.testing.assert_array_equal(crossed['p0n1'], [0, 2, 3, 5])
        for idx in crossed.values():
            self.assertEqual(len(np.unique(idx)), 4)
            self.assertEqual(y[idx].sum(), 2)

    def test_row_order_invariance(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0]); order = np.array([7, 5, 3, 1, 6, 4, 2, 0])
        e = np.array([0, 1, 2, 3]); b = np.array([2, 3, 4, 5])
        inverse = np.argsort(order)
        left = cross_memberships(y, e, b); right = cross_memberships(y[order], inverse[e], inverse[b])
        for key in left: np.testing.assert_array_equal(left[key], order[right[key]])

    def test_invalid_parents_fail(self):
        for e, b in (([0, 0], [0, 1]), ([0, 1], [0, 2]), ([0, 8], [0, 1]), ([0.0, 1.0], [0, 1])):
            with self.assertRaises(ValueError): cross_memberships([1, 0, 1, 0], e, b)

    def test_factorial_algebra(self):
        groups = {arm+'__'+mod: np.array([v, v*2]) for arm, v in zip(('p0n0', 'p1n0', 'p0n1', 'p1n1'), (1., 3., 4., 9.)) for mod in ('image', 'fusion', 'gain')}
        result = factorial_contrasts(groups)
        np.testing.assert_array_equal(result['interaction__fusion'], [3., 6.])
        np.testing.assert_array_equal(result['positive_at_n0__fusion'] + result['negative_at_p1__fusion'], result['total__fusion'])
        np.testing.assert_array_equal(result['negative_at_p0__fusion'] + result['positive_at_n1__fusion'], result['total__fusion'])


if __name__ == '__main__': unittest.main()
