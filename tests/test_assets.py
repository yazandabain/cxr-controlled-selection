"""Public aggregate replay and release metadata tests."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location('aggregate_asset_builder', ROOT / 'scripts/build_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssetTests(unittest.TestCase):
    def test_rebuild_all_tables_and_compare_retained_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'assets'
            record = builder().build(output)
            self.assertEqual(record['table_files'], 15)
            for actual in (output / 'tables').glob('*.csv'):
                expected = ROOT / 'results/tables' / actual.name
                with actual.open(encoding='utf-8', newline='') as a, expected.open(encoding='utf-8', newline='') as b:
                    self.assertEqual(list(csv.reader(a)), list(csv.reader(b)), actual.name)
            self.assertEqual(len(list((output / 'figures').glob('*.svg'))), 7)
            with self.assertRaises(FileExistsError):
                builder().build(output)

    def test_ci_and_citation_metadata(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        job = workflow['jobs']['public-release']
        self.assertEqual(job['strategy']['matrix']['os'], ['ubuntu-latest', 'windows-latest'])
        steps = job['steps']
        self.assertIn('3.13', [step.get('with', {}).get('python-version') for step in steps])
        for step in steps:
            if 'uses' in step:
                self.assertRegex(step['uses'], r'@[0-9a-f]{40}$')
        commands = '\n'.join(step.get('run', '') for step in steps)
        for required in ('pip install', 'unittest discover', 'build_assets.py', 'check_release.py'):
            self.assertIn(required, commands)
        citation = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
        self.assertEqual(citation['cff-version'], '1.2.0')
        self.assertEqual(citation['version'], '0.1.0')
        self.assertEqual(citation['type'], 'software')
        self.assertNotIn('doi', citation)
        self.assertNotIn('license', citation)


if __name__ == '__main__':
    unittest.main()
