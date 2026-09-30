"""Rebuild figures and summary tables from public aggregate files only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cxr_selection.aggregate_tables import build_core, table
from cxr_selection.control_tables import build_controls
from cxr_selection.publication_figures import build as build_figures

ROOT = Path(__file__).resolve().parents[1]


def build(out, bundle=None):
    out = Path(out).resolve()
    bundle = Path(bundle or ROOT / 'results/aggregates').resolve()
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    out.mkdir(parents=True)
    core = json.loads((bundle / 'core.json').read_text(encoding='utf-8'))
    tables = out / 'tables'
    tables.mkdir()
    build_core(core, tables)
    strength = bundle / 'strength_20260913'
    names = ('addback', 'class_source', 'pair_accounting', 'budget_comparison')
    controls = [json.loads((strength / (name + '.json')).read_text(encoding='utf-8')) for name in names]
    build_controls(*controls, tables)
    states = core['description']['panels']['train']
    rows = []
    labels = {'temperature': 'Temperature', 'heartrate': 'Heart rate', 'resprate': 'Respiratory rate',
              'o2sat': 'Oxygen saturation', 'sbp': 'Systolic pressure', 'dbp': 'Diastolic pressure',
              'pain': 'Pain', 'acuity': 'Acuity'}
    for key, label in labels.items():
        row = [label]
        for group in ('explicit', 'excluded_by_report_state'):
            value = states[group]['values'][key]
            row += [f"{value['median']:g} [{value['q25']:g}, {value['q75']:g}]",
                    f"{value['missing']:,} ({100 * value['missing'] / states[group]['studies']:.1f}%)"]
        rows.append(row)
    table(tables, 'tableS1_characteristics', ['Value', 'Explicit', 'Missing', 'Excluded', 'Missing'], rows)
    build_figures(bundle, out / 'figures')
    record = {'source_hashes': {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(bundle.rglob('*.json'))},
              'table_files': len(list(tables.glob('*.csv'))),
              'quantitative_figures': 7,
              'note': 'Presentation replay only; no model training or patient-level data access.'}
    (out / 'asset_record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out)))
