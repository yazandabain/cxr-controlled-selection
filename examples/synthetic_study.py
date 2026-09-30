"""Small synthetic demonstration. No patient data are read or written."""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from cxr_selection.evaluate import CachedMetric
from cxr_selection.models import fit_logistic, sample_training_panels
from cxr_selection.refit_sensitivity import patient_weights


def synthetic_partition(rng, name, patients, repeats=1):
    """Artificial patient features; repeated rows share the same binary target."""
    base = rng.normal(size=(patients, 6))
    target = (base[:, 0] + .6 * base[:, 3] + rng.normal(size=patients) > 0).astype(int)
    ids = np.array([f"synthetic-{name}-{i:04d}" for i in range(patients)])
    features = np.repeat(base, repeats, axis=0) + rng.normal(scale=.1, size=(patients * repeats, 6))
    return {"ids": np.repeat(ids, repeats), "X": features, "y": np.repeat(target, repeats)}


def run_demo(seed=7, draws=200):
    if draws < 2:
        raise ValueError('At least two bootstrap draws are required')
    rng = np.random.default_rng(seed)
    train = synthetic_partition(rng, 'train', 768)
    tune = synthetic_partition(rng, 'validation', 256)
    test = synthetic_partition(rng, 'test', 256, repeats=2)
    assert not set(train['ids']) & set(tune['ids'])
    assert not set(train['ids']) & set(test['ids'])
    assert not set(tune['ids']) & set(test['ids'])

    # Eligibility is a different variable from the reference target.
    states = rng.choice(np.array([0., 1., -1., np.nan]), size=len(train['y']))
    panels = sample_training_panels(train['ids'], states, train['y'], (64, 128), (0, 1))
    # One common training-only unlabeled transform, shared across every arm.
    scaler = StandardScaler().fit(train['X'])
    x = {name: scaler.transform(part['X']) for name, part in
         [('train', train), ('validation', tune), ('test', test)]}
    cache = {}
    for panel_seed, budgets in panels.items():
        for budget, arms in budgets.items():
            for arm, indices in arms.items():
                assert len(indices) == len(np.unique(train['ids'][indices])) == budget
                for modality, columns in [('image_like', slice(0, 3)), ('combined', slice(0, 6))]:
                    fit = fit_logistic(x['train'][indices, columns], train['y'][indices],
                                       x['validation'][:, columns], tune['y'], x['test'][:, columns])
                    key = (budget, arm, modality)
                    cache.setdefault(key, []).append(CachedMetric(test['y'], fit['test_probabilities']))

    def statistics(weights):
        # Mean of per-fit metrics, not the metric of averaged predictions.
        values = {key: np.mean([metric(weights) for metric in metrics], axis=0)
                  for key, metrics in cache.items()}
        for budget in (64, 128):
            for arm in ('explicit', 'random', 'prevalence_matched'):
                values[(budget, arm, 'combined_minus_image_like')] = (
                    values[(budget, arm, 'combined')] - values[(budget, arm, 'image_like')])
            values[(budget, 'matched_minus_explicit', 'combined')] = (
                values[(budget, 'prevalence_matched', 'combined')] - values[(budget, 'explicit', 'combined')])
        return values

    point = statistics(np.ones(len(test['y'])))
    boot = {key: [] for key in point}
    for _ in range(draws):
        values = statistics(patient_weights(test['ids'], rng))
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Degenerate bootstrap draw; do not silently discard it')
        for key, value in values.items():
            boot[key].append(value)
    results = {}
    for key, value in point.items():
        interval = np.quantile(boot[key], [.025, .975], axis=0)
        results['__'.join(map(str, key))] = {
            metric: {'point': float(value[j]), 'ci95': interval[:, j].tolist()}
            for j, metric in enumerate(('auroc', 'average_precision'))}
    return {'data': 'synthetic only', 'seed': seed, 'draws': draws,
            'uncertainty': 'patient-paired; conditional on fixed models and panels',
            'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--draws', type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.seed, args.draws), indent=2))
