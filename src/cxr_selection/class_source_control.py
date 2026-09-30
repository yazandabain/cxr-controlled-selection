"""Cross positive and negative membership sources."""
import numpy as np


def cross_memberships(y, explicit, broad):
    y = np.asarray(y)
    parents = [np.asarray(v) for v in (explicit, broad)]
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('Binary coded labels required')
    for v in parents:
        if v.ndim != 1 or not np.issubdtype(v.dtype, np.integer) or (v < 0).any() or (v >= len(y)).any() or len(np.unique(v)) != len(v):
            raise ValueError('Unique valid parent indices required')
    e, b = parents
    if len(e) != len(b) or y[e].sum() != y[b].sum():
        raise ValueError('Parent class counts must match exactly')
    if not 0 < y[e].sum() < len(e):
        raise ValueError('Both coded classes required')
    result = {'p0n0': e.copy(), 'p1n1': b.copy(),
              'p1n0': np.r_[b[y[b] == 1], e[y[e] == 0]],
              'p0n1': np.r_[e[y[e] == 1], b[y[b] == 0]]}
    for idx in result.values():
        assert len(idx) == len(e) == len(np.unique(idx))
        assert y[idx].sum() == y[e].sum()
    return result


def factorial_contrasts(groups):
    result = {}
    for mod in ('image', 'fusion', 'gain'):
        a, b, c, d = (groups[k + '__' + mod] for k in ('p0n0', 'p1n0', 'p0n1', 'p1n1'))
        for name, value in (
            ('positive_at_n0', b-a), ('positive_at_n1', d-c),
            ('negative_at_p0', c-a), ('negative_at_p1', d-b),
            ('interaction', d-b-c+a), ('total', d-a)):
            result[name + '__' + mod] = value
    return result
