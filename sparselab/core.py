"""Cyclic LASSO descent and explicit floating-point optimality witnesses."""
import math
import sys


def number(v, name, low=-100.0, high=100.0):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(name + ' must be a finite number')
    try:
        v = float(v)
    except OverflowError as exc:
        raise ValueError(name + ' is out of range') from exc
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(name + ' is out of range')
    return v


def validate(x, y):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 128:
        raise ValueError('X needs 1..128 rows')
    if not isinstance(x[0], (list, tuple)) or not 1 <= len(x[0]) <= 12:
        raise ValueError('X needs 1..12 columns')
    p = len(x[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != p for row in x):
        raise ValueError('X must be rectangular')
    if not isinstance(y, (list, tuple)) or len(y) != len(x):
        raise ValueError('y must match X')
    x = [[number(v, 'X entry') for v in row] for row in x]
    for j in range(p):
        norm = math.fsum(row[j] ** 2 for row in x) / len(x)
        if any(row[j] != 0 for row in x) and norm < 1e-8:
            raise ValueError('nonzero column mean squared norm must be at least 1e-8')
    return x, [number(v, 'y entry') for v in y]


def dot(a, b):
    return math.fsum(u * v for u, v in zip(a, b))


def soft(value, threshold):
    return math.copysign(max(abs(value) - threshold, 0.0), value) if abs(value) > threshold else 0.0


def _witness(x, y, beta, penalty):
    n, p = len(x), len(x[0])
    prediction = [dot(row, beta) for row in x]
    residual = [yi - fi for yi, fi in zip(y, prediction)]
    correlations = [math.fsum(row[j] * r for row, r in zip(x, residual)) / n for j in range(p)]
    loss = dot(residual, residual) / (2 * n)
    regularizer = penalty * math.fsum(abs(v) for v in beta)
    primal = loss + regularizer
    maxcor = max(abs(c) for c in correlations)
    # Interior scaling is intentional: a floating-point witness, not an interval proof.
    scale = min(1.0, penalty / maxcor) if maxcor else 1.0
    scale *= 1 - 16 * sys.float_info.epsilon
    for _ in range(8):
        dual_vector = [scale * r / n for r in residual]
        dual_correlations = [math.fsum(row[j] * u for row, u in zip(x, dual_vector)) for j in range(p)]
        if max(abs(c) for c in dual_correlations) <= penalty:
            break
        scale *= 0.5
    else:
        raise ArithmeticError('unable to construct feasible dual witness')
    dual = dot(y, dual_vector) - n * dot(dual_vector, dual_vector) / 2
    # Algebraically P-D, avoiding cancellation between two large objectives.
    gap_terms = [penalty * abs(b) - b * c for b, c in zip(beta, dual_correlations)]
    mismatch = [r - n * u for r, u in zip(residual, dual_vector)]
    gap = dot(mismatch, mismatch) / (2 * n) + math.fsum(gap_terms)
    kkt = [abs(c - math.copysign(penalty, b)) if b else max(abs(c) - penalty, 0.0)
           for b, c in zip(beta, correlations)]
    return dict(coefficients=list(beta), prediction=prediction, residual=residual,
                correlations=correlations, squared_loss=loss, l1_penalty=regularizer,
                primal=primal, dual=dual, dual_vector=dual_vector, dual_scale=scale,
                dual_correlations=dual_correlations, dual_feasibility_max=max(abs(c) for c in dual_correlations),
                gap=gap, direct_gap=primal - dual, gap_terms=gap_terms,
                max_kkt=max(kkt), kkt_residuals=kkt, active=[j for j, b in enumerate(beta) if b != 0.0])


def witness(x, y, coefficients, penalty):
    x, y = validate(x, y)
    penalty = number(penalty, 'penalty', 1e-8, 10000)
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(x[0]):
        raise ValueError('coefficients must match columns')
    beta = [number(b, 'coefficient', -1e6, 1e6) for b in coefficients]
    return _witness(x, y, beta, penalty)


def fit(x, y, penalty, *, tolerance=1e-10, max_sweeps=20000, initial=None, order=None):
    """No implicit intercept or standardization; all supplied columns penalized."""
    x, y = validate(x, y)
    penalty = number(penalty, 'penalty', 1e-8, 10000)
    tolerance = number(tolerance, 'tolerance', 1e-12, 1e-5)
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or not 1 <= max_sweeps <= 50000:
        raise ValueError('max_sweeps must be an integer in 1..50000')
    n, p = len(x), len(x[0])
    if order is None:
        order = list(range(p))
    if (not isinstance(order, (list, tuple)) or len(order) != p or
        any(isinstance(j, bool) or not isinstance(j, int) for j in order) or sorted(order) != list(range(p))):
        raise ValueError('order must be a permutation of column indices')
    if initial is None:
        beta = [0.0] * p
    else:
        if not isinstance(initial, (list, tuple)) or len(initial) != p:
            raise ValueError('initial must match columns')
        beta = [number(v, 'initial', -1e6, 1e6) for v in initial]
    norms = [math.fsum(row[j] ** 2 for row in x) / n for j in range(p)]
    history = []
    current = _witness(x, y, beta, penalty)
    history.append(dict(sweep=0, primal=current['primal'], gap=current['gap'], max_kkt=current['max_kkt']))
    converged = current['gap'] >= 0 and current['gap'] <= tolerance and current['max_kkt'] <= tolerance
    sweeps = 0
    while not converged and sweeps < max_sweeps:
        for j in order:
            if norms[j] == 0:
                beta[j] = 0.0
                continue
            # Recompute partial residuals: no accumulated residual drift.
            partial = [yi - math.fsum(row[k] * beta[k] for k in range(p) if k != j) for row, yi in zip(x, y)]
            rho = math.fsum(row[j] * r for row, r in zip(x, partial)) / n
            beta[j] = soft(rho, penalty) / norms[j]
        sweeps += 1
        current = _witness(x, y, beta, penalty)
        history.append(dict(sweep=sweeps, primal=current['primal'], gap=current['gap'], max_kkt=current['max_kkt']))
        converged = current['gap'] >= 0 and current['gap'] <= tolerance and current['max_kkt'] <= tolerance
    return dict(schema_version=1, objective='||y-X beta||^2/(2N) + penalty*||beta||_1',
                n=n, p=p, penalty=penalty, tolerance=tolerance, max_sweeps=max_sweeps,
                converged=converged, status='converged' if converged else 'sweep_limit',
                sweeps=sweeps, order=list(order), initial=[0.0] * p if initial is None else list(initial),
                history=history, **current)


def path(x, y, penalties, *, tolerance=1e-10, max_sweeps=20000):
    x, y = validate(x, y)
    if not isinstance(penalties, (list, tuple)) or not 1 <= len(penalties) <= 32:
        raise ValueError('penalties needs 1..32 descending distinct values')
    penalties = [number(v, 'penalty', 1e-8, 10000) for v in penalties]
    if any(a <= b for a, b in zip(penalties, penalties[1:])):
        raise ValueError('penalties must strictly decrease')
    results, beta = [], None
    for penalty in penalties:
        result = fit(x, y, penalty, tolerance=tolerance, max_sweeps=max_sweeps, initial=beta)
        results.append(result)
        beta = result['coefficients']
    lambda_max = max(abs(math.fsum(row[j] * yi for row, yi in zip(x, y)) / len(x)) for j in range(len(x[0])))
    return dict(schema_version=1, X=x, y=y, lambda_max=lambda_max, penalties=penalties,
                warm_started=True, all_converged=all(r['converged'] for r in results), results=results)
