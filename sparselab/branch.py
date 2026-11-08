"""Exact affine KKT branch certificates, not a general path-following solver."""
from fractions import Fraction as F
from .core import validate
from .oracle import _solve


def branch(x, y, signs):
    """Maximal closed lambda>=0 interval for one weak active/sign branch.

    Selected signs impose s_j*beta_j>=0, including boundary zeros. A singular
    selected Gram returns an explicit noncertificate; no inverse is guessed.
    """
    x, y = validate(x, y)
    n, p = len(x), len(x[0])
    if (not isinstance(signs, (list, tuple)) or len(signs) != p or
            any(type(s) is not int or s not in (-1, 0, 1) for s in signs)):
        raise ValueError('signs must match columns and contain integer -1, 0, 1')
    x = [[F(v) for v in row] for row in x]
    y = list(map(F, y))
    zero = F(0)
    gram = [[sum((r[j]*r[k] for r in x), zero)/n for k in range(p)] for j in range(p)]
    cross = [sum((r[j]*v for r, v in zip(x, y)), zero)/n for j in range(p)]
    active = [j for j, s in enumerate(signs) if s]
    matrix = [[gram[j][k] for k in active] for j in active]
    intercept = _solve(matrix, [cross[j] for j in active])
    direction = _solve(matrix, [F(signs[j]) for j in active])
    def strings(v):
        return [str(t) for t in v]
    report = dict(schema_version=1, arithmetic='exact rational representation of validated binary64 inputs',
                  scope='one weak active/sign branch; not a general path-following solver',
                  X=[strings(r) for r in x], y=strings(y), signs=list(signs), selected=active,
                  gram=[strings(r) for r in gram], cross=strings(cross))
    if intercept is None or direction is None:
        return dict(report, status='singular_active_gram', certified=False,
                    reason='selected Gram is singular; no ridge, pseudoinverse or full-path claim')
    a, d = [zero]*p, [zero]*p
    for j, aj, dj in zip(active, intercept, direction):
        a[j], d[j] = aj, -dj
    # beta_j(lambda)=a_j+lambda*d_j; correlations have the same affine form.
    c0 = [cross[j]-sum((gram[j][k]*a[k] for k in range(p)), zero) for j in range(p)]
    c1 = [-sum((gram[j][k]*d[k] for k in range(p)), zero) for j in range(p)]
    constraints = [dict(kind='nonnegative_lambda', feature=None, intercept='0', slope='1')]
    for j in range(p):
        if signs[j]:
            u, v = signs[j]*a[j], signs[j]*d[j]
            constraints.append(dict(kind='selected_sign', feature=j, intercept=str(u), slope=str(v)))
            assert c0[j] == 0 and c1[j] == signs[j]
        else:
            constraints.extend([
                dict(kind='inactive_upper', feature=j, intercept=str(-c0[j]), slope=str(1-c1[j])),
                dict(kind='inactive_lower', feature=j, intercept=str(c0[j]), slope=str(1+c1[j]))])
    lower, upper, impossible = zero, None, []
    for i, r in enumerate(constraints):
        u, v = F(r['intercept']), F(r['slope'])
        if v > 0:
            bound = -u/v
            lower = max(lower, bound)
            r.update(bound_type='lower', bound=str(bound))
        elif v < 0:
            bound = -u/v
            upper = bound if upper is None else min(upper, bound)
            r.update(bound_type='upper', bound=str(bound))
        else:
            r.update(bound_type='constant', bound=None)
            if u < 0:
                impossible.append(i)
    report.update(coefficient_intercepts=strings(a), coefficient_slopes=strings(d),
                  correlation_intercepts=strings(c0), correlation_slopes=strings(c1),
                  constraints=constraints, lower=str(lower), upper=None if upper is None else str(upper),
                  impossible_constraints=impossible)
    if impossible or (upper is not None and lower > upper):
        return dict(report, status='infeasible_sign_pattern', certified=False)
    def witness(lam):
        beta = [v+lam*w for v, w in zip(a, d)]
        pred = [sum((v*b for v, b in zip(r, beta)), zero) for r in x]
        residual = [v-f for v, f in zip(y, pred)]
        corr = [v+lam*w for v, w in zip(c0, c1)]
        slacks = [F(r['intercept'])+lam*F(r['slope']) for r in constraints]
        assert all(s >= 0 for s in slacks)
        assert all(c == (lam if b > 0 else -lam) if b else abs(c) <= lam for b, c in zip(beta, corr))
        dual_vector = [v/n for v in residual]
        primal = sum((r*r for r in residual), zero)/(2*n)+lam*sum((abs(b) for b in beta), zero)
        dual = sum((v*u for v, u in zip(y, dual_vector)), zero)-n*sum((u*u for u in dual_vector), zero)/2
        assert primal == dual
        return dict(penalty=str(lam), coefficients=strings(beta), prediction=strings(pred),
                    residual=strings(residual), correlations=strings(corr), dual_vector=strings(dual_vector),
                    primal=str(primal), dual=str(dual), gap=str(primal-dual), slacks=strings(slacks),
                    tight_constraints=[i for i, s in enumerate(slacks) if s == 0])
    probe = lower+1 if upper is None else (lower+upper)/2
    probes = [witness(lower)]
    if upper is not None and upper != lower:
        probes.append(witness(upper))
    report.update(status='certified_point' if upper == lower else 'certified_segment', certified=True,
                  endpoints=probes, interior=witness(probe),
                  strict_selected_signs_at_probe=all(signs[j]*(a[j]+probe*d[j]) > 0 for j in active))
    return report


def branch_demo():
    a, b = [1, 1, -1, -1], [1, -1, 1, -1]
    x = [[u, 2*u+v] for u, v in zip(a, b)]
    y = [2*u-v for u, v in zip(a, b)]
    patterns = [[0, 0], [0, 1], [1, 1], [1, 0], [1, -1]]
    segments = [branch(x, y, s) for s in patterns]
    assert [(r['lower'], r['upper']) for r in segments] == [
        ('3', None), ('4/3', '3'), ('1', '4/3'), ('1/3', '1'), ('0', '1/3')]
    for left, right in zip(segments, segments[1:]):
        assert left['endpoints'][0]['coefficients'] == right['endpoints'][-1]['coefficients']
    assert segments[-1]['endpoints'][0]['coefficients'] == ['4', '-1']
    singular = branch([[u, u] for u in a], [2*u for u in a], [1, 1])
    duplicate_left = branch([[u, u] for u in a], [2*u for u in a], [1, 0])
    duplicate_right = branch([[u, u] for u in a], [2*u for u in a], [0, 1])
    assert not singular['certified']
    assert duplicate_left['interior']['prediction'] == duplicate_right['interior']['prediction']
    return dict(schema_version=1, scope='developed supplied fixtures, not causal feature selection',
                correlated=dict(X=x, y=y, segments=segments,
                                events=[dict(penalty='3', feature=1, event='enter_positive'),
                                        dict(penalty='4/3', feature=0, event='enter_positive'),
                                        dict(penalty='1', feature=1, event='leave'),
                                        dict(penalty='1/3', feature=1, event='reenter_negative')]),
                infeasible=branch(x, y, [-1, 1]),
                singular=singular, duplicate_branches=[duplicate_left, duplicate_right],
                singleton=branch([[1, 1]], [0], [1, 0]))
