"""Independent exact-binary-rational active/sign enumeration, at most 5 columns."""
from fractions import Fraction
from itertools import combinations, product
from .core import number, validate


def _solve(a, b):
    size = len(b)
    aug = [list(row) + [v] for row, v in zip(a, b)]
    for j in range(size):
        pivot = next((i for i in range(j, size) if aug[i][j]), None)
        if pivot is None:
            return None
        aug[j], aug[pivot] = aug[pivot], aug[j]
        divisor = aug[j][j]
        aug[j] = [v / divisor for v in aug[j]]
        for i in range(size):
            if i != j:
                multiplier = aug[i][j]
                aug[i] = [u - multiplier * v for u, v in zip(aug[i], aug[j])]
    return [row[-1] for row in aug]


def enumerate_optima(x, y, penalty):
    x, y = validate(x, y)
    penalty = number(penalty, 'penalty', 1e-8, 10000)
    n, p = len(x), len(x[0])
    if n > 32 or p > 5:
        raise ValueError('exact oracle limited to 32 rows and 5 columns')
    x = [[Fraction(v) for v in row] for row in x]
    y = [Fraction(v) for v in y]
    lam = Fraction(penalty)
    zero = Fraction(0)
    gram = [[sum((row[j] * row[k] for row in x), zero) / n for k in range(p)] for j in range(p)]
    cross = [sum((row[j] * yi for row, yi in zip(x, y)), zero) / n for j in range(p)]
    accepted, considered, singular = [], 0, 0
    for size in range(p + 1):
        for subset in combinations(range(p), size):
            a = [[gram[j][k] for k in subset] for j in subset]
            for signs in product((-1, 1), repeat=size):
                considered += 1
                vals = _solve(a, [cross[j] - lam * s for j, s in zip(subset, signs)])
                if vals is None:
                    singular += 1
                    continue
                if any(v * s <= 0 for v, s in zip(vals, signs)):
                    continue
                beta = [zero] * p
                for j, v in zip(subset, vals):
                    beta[j] = v
                corr = [cross[j] - sum((gram[j][k] * beta[k] for k in range(p)), zero) for j in range(p)]
                if any((c != (lam if b > 0 else -lam)) if b else abs(c) > lam for c, b in zip(corr, beta)):
                    continue
                prediction = [sum((v * b for v, b in zip(row, beta)), zero) for row in x]
                residual = [yi - f for yi, f in zip(y, prediction)]
                objective = sum((v * v for v in residual), zero) / (2 * n) + lam * sum((abs(v) for v in beta), zero)
                accepted.append(dict(coefficients=[str(v) for v in beta], prediction=[str(v) for v in prediction],
                                     primal=str(objective), correlations=[str(v) for v in corr], active=list(subset)))
    if not accepted:
        raise ArithmeticError('no exact KKT witness found')
    return dict(schema_version=1, arithmetic='exact rational representation of validated binary64 inputs',
                considered=considered, singular_sign_patterns=singular, witnesses=accepted,
                witness_count=len(accepted), objectives_agree=len({r['primal'] for r in accepted}) == 1,
                predictions_agree=len({tuple(r['prediction']) for r in accepted}) == 1)
