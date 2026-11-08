"""Developed analytic controls, not a benchmark or data-generating claim."""
from .core import fit, path, witness
from .oracle import enumerate_optima


def demo():
    a, b, c = [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]
    x = [list(row) for row in zip(a, b, c)]
    y = [2 * u - .75 * v + .125 * w for u, v, w in zip(a, b, c)]
    regularization_path = path(x, y, [2, 1, .5, .25, .125, .0625])
    exact_path = [enumerate_optima(x, y, lam) for lam in regularization_path['penalties']]
    duplicate_x = [list(row) for row in zip(a, a, b)]
    duplicate_y = [2 * u - .75 * v for u, v in zip(a, b)]
    forward = fit(duplicate_x, duplicate_y, .25)
    reverse = fit(duplicate_x, duplicate_y, .25, order=[2, 1, 0])
    midpoint = witness(duplicate_x, duplicate_y, [.875, .875, -.5], .25)
    duplicate_oracle = enumerate_optima(duplicate_x, duplicate_y, .25)
    # Early-stopping negative control is not a modified objective or weak baseline.
    correlated_x = [list(row) for row in zip(a, [u + .25 * v for u, v in zip(a, b)])]
    correlated_y = [2 * u + .75 * v for u, v in zip(a, b)]
    limited = fit(correlated_x, correlated_y, .125, max_sweeps=1)
    converged = fit(correlated_x, correlated_y, .125)
    correlated_oracle = enumerate_optima(correlated_x, correlated_y, .125)
    assert regularization_path['all_converged'] and forward['converged'] and reverse['converged'] and converged['converged']
    assert not limited['converged'] and limited['gap'] > 1e-4
    assert forward['coefficients'] != reverse['coefficients']
    assert max(abs(u - v) for u, v in zip(forward['prediction'], reverse['prediction'])) < 1e-10
    assert duplicate_oracle['witness_count'] == 2 and duplicate_oracle['predictions_agree']
    return dict(schema_version=1, scope='developed small supplied regression fixtures, no causal feature claim',
                orthogonal=dict(path=regularization_path, exact=exact_path),
                duplicate=dict(X=duplicate_x, y=duplicate_y, penalty=.25, forward=forward, reverse=reverse,
                               midpoint=midpoint, exact=duplicate_oracle),
                correlated=dict(X=correlated_x, y=correlated_y, penalty=.125, limited=limited,
                                converged=converged, exact=correlated_oracle))
