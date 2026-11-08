"""Exact-rational Gap Safe sphere audits for the stored-input LASSO problem."""
from fractions import Fraction
from .core import number, validate
from .oracle import enumerate_optima


def screen(x, y, coefficients, penalty):
    """Certify safe zeros without square roots or rounded comparisons.

    This audits supplied iterates. It does not change the floating solver.
    """
    x, y = validate(x, y)
    lam = Fraction(number(penalty, 'penalty', 1e-8, 10000))
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(x[0]):
        raise ValueError('coefficients must match columns')
    beta = [Fraction(number(v, 'coefficient', -1e6, 1e6)) for v in coefficients]
    x = [[Fraction(v) for v in row] for row in x]
    y = [Fraction(v) for v in y]
    n, p, zero = len(x), len(x[0]), Fraction(0)
    prediction = [sum((a * b for a, b in zip(row, beta)), zero) for row in x]
    residual = [yi - fi for yi, fi in zip(y, prediction)]
    corr = [sum((row[j] * r for row, r in zip(x, residual)), zero) / n for j in range(p)]
    maxcor = max(abs(c) for c in corr)
    scale = min(Fraction(1), lam / maxcor) if maxcor else Fraction(1)
    u = [scale * r / n for r in residual]
    dual_corr = [sum((row[j] * ui for row, ui in zip(x, u)), zero) for j in range(p)]
    assert max(abs(c) for c in dual_corr) <= lam
    primal = sum((r * r for r in residual), zero) / (2 * n) + lam * sum((abs(b) for b in beta), zero)
    dual = sum((yi * ui for yi, ui in zip(y, u)), zero) - n * sum((ui * ui for ui in u), zero) / 2
    gap = primal - dual
    assert gap >= 0
    # Dual D is N-strongly concave. Its optimum is inside this sphere.
    radius_squared = 2 * gap / n
    features = []
    for j in range(p):
        norm_squared = sum((row[j] * row[j] for row in x), zero)
        margin = lam - abs(dual_corr[j])
        ball_term = norm_squared * radius_squared
        margin_squared = margin * margin
        # Squaring is equivalent only with a strictly positive unsquared margin.
        safe = margin > 0 and ball_term < margin_squared
        features.append(dict(index=j, center_correlation=str(dual_corr[j]), column_norm_squared=str(norm_squared),
                             margin=str(margin), margin_squared=str(margin_squared), ball_term=str(ball_term),
                             strict_slack=str(margin_squared - ball_term), safe_zero=safe,
                             reason='strict_ball_test' if safe else 'not_certified'))
    return dict(schema_version=1, arithmetic='exact rational representation of validated binary64 inputs',
                objective='||y-X beta||^2/(2N) + penalty*||beta||_1', n=n, p=p,
                X=[[str(v) for v in row] for row in x], y=[str(v) for v in y],
                coefficients=[str(v) for v in beta], penalty=str(lam), prediction=[str(v) for v in prediction],
                residual=[str(v) for v in residual], residual_correlations=[str(v) for v in corr],
                primal=str(primal), dual=str(dual), gap=str(gap), dual_scale=str(scale),
                dual_vector=[str(v) for v in u], dual_correlations=[str(v) for v in dual_corr],
                dual_feasible=True, radius_squared=str(radius_squared), features=features,
                safe_zero=[f['index'] for f in features if f['safe_zero']],
                not_certified=[f['index'] for f in features if not f['safe_zero']],
                conclusion='safe_zero entries vanish in every optimizer of this exact stored-input objective; not_certified is not evidence of activity')


def screen_demo():
    a,b,c=[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]
    x=[list(row) for row in zip(a,b,c)]
    y=[2*u-.75*v+.125*w for u,v,w in zip(a,b,c)]
    duplicate_x=[list(row) for row in zip(a,a,b)]
    duplicate_y=[2*u-.75*v for u,v in zip(a,b)]
    controls=[('Loose iterate',x,y,[0,0,0],.25),
              ('Near, nonzero inactive',x,y,[1.75,-.5,.03125],.25),
              ('Exact interior zero',x,y,[1.75,-.5,0],.25),
              ('Exact boundary zero',x,y,[1.875,-.625,0],.125),
              ('Duplicate ambiguity',duplicate_x,duplicate_y,[1.75,0,-.5],.25),
              ('Zero column',[[1,0],[1,0]],[2,2],[0,3],.25)]
    results=[]
    for name,xx,yy,beta,lam in controls:
        result=screen(xx,yy,beta,lam)
        exact=enumerate_optima(xx,yy,lam)
        assert all(Fraction(w['coefficients'][j])==0 for j in result['safe_zero'] for w in exact['witnesses'])
        results.append(dict(name=name,screen=result,exact=exact))
    assert [r['screen']['safe_zero'] for r in results]==[[],[2],[2],[],[],[1]]
    assert results[1]['screen']['gap']=='5/1024'
    assert results[2]['screen']['gap']==results[3]['screen']['gap']=='0'
    assert results[4]['exact']['witness_count']==2
    return dict(schema_version=1, scope='exact stored-input Gap Safe audits, not solver acceleration or causal screening', controls=results)
