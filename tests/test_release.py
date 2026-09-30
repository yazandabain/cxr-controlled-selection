"""Check the source-only public boundary and import behavior."""
import importlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


def checker():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('public_release_check', root / 'scripts/check_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TagTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identity = ['-c', 'user.name=Synthetic tester', '-c',
                         'user.email=' + 'tester' + '@' + 'users.noreply.github.com']
        self.run_git('init')
        self.run_git('commit', '--allow-empty', '-m', 'synthetic fixture')

    def run_git(self, *args, identity=None):
        result = subprocess.run(
            ['git', '-c', f'safe.directory={self.root.as_posix()}',
             *(self.identity if identity is None else identity), '-C', str(self.root), *args],
            check=True, capture_output=True, text=True, encoding='utf-8',
        )
        return result.stdout.strip()

    def test_annotated_tag_with_no_reply_identity(self):
        self.run_git('tag', '-a', 'v0.1.0', '-m', 'prepare release')
        result = checker().check_tags(self.root)
        self.assertEqual(result['issues'], [])
        self.assertEqual(len(result['annotated_tags']), 1)
        self.assertEqual(result['annotated_tags'][0]['target'], self.run_git('rev-parse', 'HEAD'))

    def test_personal_tagger_email_is_rejected(self):
        identity = ['-c', 'user.name=Synthetic tester', '-c',
                    'user.email=' + 'private' + '@' + 'example.invalid']
        self.run_git('tag', '-a', 'private-tagger', '-m', 'fixture', identity=identity)
        self.assertTrue(any('personal email' in issue for issue in checker().check_tags(self.root)['issues']))

    def test_credentials_and_private_paths_in_tag_message_are_rejected(self):
        message = 'gh' + 'p_' + 'A' * 36 + '\n' + 'C:' + '\\' + 'MultiModalFinal' + '\\private'
        self.run_git('tag', '-a', 'private-message', '-m', message)
        issues = checker().check_tags(self.root)['issues']
        self.assertTrue(any('credential' in issue for issue in issues))
        self.assertTrue(any('personal local path' in issue for issue in issues))

    def test_no_reply_exemption_does_not_cover_annotation_body(self):
        self.run_git('tag', '-a', 'private-body', '-m', 'tester' + '@' + 'users.noreply.github.com')
        self.assertTrue(any('personal email' in issue for issue in checker().check_tags(self.root)['issues']))

    def test_nested_tags_scan_inner_annotation_once(self):
        self.run_git('tag', '-a', 'inner', '-m', 'gh' + 'p_' + 'B' * 36)
        self.run_git('tag', '-a', 'outer', 'inner', '-m', 'outer fixture')
        self.run_git('tag', '-d', 'inner')
        result = checker().check_tags(self.root)
        self.assertEqual(len(result['annotated_tags']), 2)
        self.assertTrue(any('credential' in issue for issue in result['issues']))

    def test_annotated_and_lightweight_non_commit_targets_are_rejected(self):
        blob = subprocess.run(
            ['git', '-C', str(self.root), 'hash-object', '-w', '--stdin'], input=b'synthetic fixture',
            check=True, capture_output=True,
        ).stdout.decode().strip()
        self.run_git('tag', '-a', 'annotated-blob', blob, '-m', 'fixture')
        self.run_git('tag', 'lightweight-blob', blob)
        issues = checker().check_tags(self.root)['issues']
        self.assertTrue(any('non-commit target type' in issue for issue in issues))
        self.assertTrue(any('tag target is not a commit' in issue for issue in issues))

    def test_lightweight_commit_tag_needs_no_annotation_scan(self):
        self.run_git('tag', 'lightweight')
        self.assertEqual(checker().check_tags(self.root), {'annotated_tags': [], 'issues': []})

    def test_commit_message_is_scanned_without_identity_exemption(self):
        self.run_git('commit', '--allow-empty', '-m', 'gh' + 'p_' + 'C' * 36)
        raw = subprocess.run(
            ['git', '-C', str(self.root), 'cat-file', 'commit', 'HEAD'], check=True, capture_output=True,
        ).stdout
        self.assertTrue(any('credential' in issue for issue in checker().inspect_git_metadata(raw, 'fixture')))


class ReleaseTests(unittest.TestCase):
    def test_all_public_modules_import_without_private_inputs(self):
        modules = ('models', 'evaluate', 'addback', 'class_source_control',
                   'pair_accounting', 'budget_comparison', 'group_contrast',
                   'refit_sensitivity', 'target_transfer')
        for name in modules:
            importlib.import_module('cxr_selection.' + name)

    def test_release_inventory(self):
        root = Path(__file__).resolve().parents[1]
        result = checker().check(root, check_git=False)
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
