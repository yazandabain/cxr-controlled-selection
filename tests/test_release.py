"""Check the source-only public boundary and import behavior."""
import importlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np


class ReleaseTests(unittest.TestCase):
    def test_all_public_modules_import_without_private_inputs(self):
        modules = ('models', 'evaluate', 'addback', 'class_source_control',
                   'pair_accounting', 'budget_comparison', 'group_contrast',
                   'refit_sensitivity', 'target_transfer')
        for name in modules:
            importlib.import_module('cxr_selection.' + name)

    def test_release_inventory(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('public_release_check', root / 'scripts/check_release.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.check(root, check_git=False)
        self.assertEqual(result['issues'], [])

    def test_example_partitions_are_artificial_and_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('synthetic_example', root / 'examples/synthetic_study.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(3)
        left = module.synthetic_partition(rng, 'train', 12)
        right = module.synthetic_partition(rng, 'test', 6, repeats=2)
        self.assertFalse(set(left['ids']) & set(right['ids']))
        self.assertTrue(all(str(i).startswith('synthetic-') for i in right['ids']))
        self.assertEqual(len(np.unique(right['ids'])), 6)
        np.testing.assert_array_equal(right['y'][::2], right['y'][1::2])


if __name__ == '__main__':
    unittest.main()
