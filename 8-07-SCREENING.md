# Exact Gap Safe screening audits

This extension certifies some zero coefficients for the **exact rational objective represented by stored binary64 inputs**. It does not turn the original floating-point solver into an interval implementation. It audits a supplied iterate; it does not skip coordinates, change the fit, or claim a speedup.

Standard Gap Safe sphere methods follow Fercoq, Gramfort and Salmon, [Mind the duality gap: safer rules for the Lasso](https://proceedings.mlr.press/v37/fercoq15.html), especially the sphere test and gap-radius relation (9), (17), (18) in the [paper](https://proceedings.mlr.press/v37/fercoq15.pdf). SparseLab uses the explicit mean-loss normalization below and exact rational, square-root-free comparisons. This is an implementation of a standard rule, not a new screening theorem.

## Objective and exactly feasible center

The objective is unchanged:

`P(beta) = ||y-X beta||^2/(2N) + lambda*||beta||_1`, lambda>0.

The unscaled dual is `D(u) = y^T u - (N/2)||u||^2`, subject to `max_j |X_j^T u| <= lambda`.

Validate inputs using the existing finite/bounded float model, then convert every X, y, coefficient and lambda with `Fraction(float)`. This represents each stored binary64 value exactly. In particular 0.1 is **not** rational 1/10. The theorem concerns these stored inputs, not unknown pre-rounding data or intended decimal values. No rational approximations, directed rounding, float square roots or tolerance-based decisions are used after that conversion.

For the supplied beta, compute exact residual r and correlations `c_j=X_j^T r/N`. Set `s=min(1,lambda/max_j |c_j|)`, using 1 when every c_j is zero. The exact dual point is `u=s*r/N`. All X^T u correlations are recomputed and exactly checked feasible. There is no epsilon shrink or rounded feasibility test in this extension. Exact P, D and gap `G=P-D>=0` are retained as rational strings.

## Why the sphere is safe

D is N-strongly concave. At its constrained maximizer u*, the feasible first-order inequality gives `grad D(u*) dot (u-u*) <= 0`. Therefore

`D(u) <= D(u*) - (N/2)||u-u*||^2 <= P(beta) - (N/2)||u-u*||^2`.

It follows that `||u-u*||^2 <= R_squared = 2G/N`. This N normalization matters: the paper's lambda-scaled, unnormalized dual coordinates use a different-looking radius. The report returns our actual unscaled u, not the paper's theta.

For column j, Cauchy-Schwarz bounds `|X_j^T u*| <= |X_j^T u| + ||X_j||*R`. LASSO KKT says any optimizer with nonzero coefficient j must satisfy `|X_j^T u*|=lambda`. So if the sphere bound is strictly below lambda, coefficient j must be zero in **every** optimizer, even when the coefficients are nonunique.

Define exact margin `m_j=lambda-|X_j^T u|`, exact `q_j=||X_j||^2` (sum, not mean), and exact `R_squared=2G/N`. The square-root-free test is

`m_j > 0 AND q_j*R_squared < m_j^2`.

Both comparisons are strict and rational. The positive-margin check is needed before squaring. Equality is not sufficient. Each feature retains its center correlation, norm squared, margin, squared margin, ball term and signed strict slack `m_j^2-q_j*R_squared`. `safe_zero` lists successful indices. `not_certified` means only that this sufficient test failed, **not** that a feature is active, useful or relevant.

## Six developed controls

The original four-row Hadamard columns a,b,c and the original duplicate-feature control are reused, not changed to improve a baseline.

| Control | Exact gap | R squared | safe_zero |
|---|---:|---:|---|
| Orthogonal loose beta `[0,0,0]`, lambda=1/4 | 14357/8192 | 14357/16384 | [] |
| Near beta `[7/4,-1/2,1/32]`, lambda=1/4 | 5/1024 | 5/2048 | [2] |
| Exact interior beta `[7/4,-1/2,0]`, lambda=1/4 | 0 | 0 | [2] |
| Exact boundary beta `[15/8,-5/8,0]`, lambda=1/8 | 0 | 0 | [] |
| Duplicate beta `[7/4,0,-1/2]`, lambda=1/4 | 0 | 0 | [] |
| Zero-column design `[[1,0],[1,0]]`, y=`[2,2]`, beta=`[0,3]`, lambda=1/4 | 73/32 | 73/32 | [1] |

For the near iterate, beta_2 is currently nonzero (`1/32`), but the certificate proves its optimum value must be zero. The feature-2 ball term is `5/512`, margin squared `25/1024`, strict slack `15/1024 > 0`. This is a useful certificate before exact convergence, not a check that simply reads the current support.

For the exact boundary control, beta_2 is zero and the fit is the unique orthogonal optimum, but its dual correlation equals lambda. Both margin and radius are zero, so the required strict inequality fails. Even zero gap need not screen every true zero. For duplicate features, either copy can carry nonzero optimal coefficient; neither is safely discarded, despite one being zero in the chosen sparse witness. A zero column, by contrast, has norm zero and is safely discarded even with a loose gap and a nonzero current coefficient.

A separate strict-equality test uses X=`[[1]]`, y=`[3/4]`, beta=`[1/4]`, lambda=1. Its exact gap is 1/8, radius squared 1/4, margin squared 1/4 and ball term 1/4. The feature's unique optimum is zero, yet this ball test must return not_certified. Replacing `<` with `<=` would falsely call the sphere test strict.

Every demo control also carries the independent exact active/sign enumerator's KKT witnesses. Tests check that all reported safe-zero indices vanish in those witnesses and that each corresponding exact dual optimum is inside the returned ball. The mathematical sufficient-rule proof, rather than finite testing, supplies the all-optimizers claim.

## API and CLI

```python
from sparselab import screen, screen_demo

certificate = screen(X, y, coefficients, penalty=0.25)
print(certificate['safe_zero'], certificate['gap'], certificate['radius_squared'])
print(certificate['features'])
```

```bash
python -m sparselab screen-demo
python -m sparselab screen examples/screen-input.json
python examples/render_screening.py --check
```

`screen` uses general fit input limits (<=128 rows, <=12 columns), existing bounded finite numeric/column-norm/positive-penalty checks, and audited coefficients in [-1e6,1e6]. It does not inherit the oracle's stricter <=32-row/5-column limit; only the developed demo's independent enumerations use that oracle. No optimizer is called by `screen` and no convergence requirement is imposed on the supplied iterate. Inputs are copied and global RNG state is untouched.

Strict JSON expects exactly `schema_version`, `X`, `y`, `coefficients`, `penalty`, with integer schema_version=1. Existing UTF-8, 100000-byte input cap, duplicate-key, nonfinite, bool and unknown/missing-field rejection remain. `screen-demo` has no input file. All fractions in the report are exact rational strings; dimensions, indices, booleans and schema metadata retain their natural JSON types. Full input rationals and witnesses can make output much larger than input.

## Verification and limits

The extension adds 14 behavior checks and 3 SVG checks, bringing the total from 28 to 45. Tests reconstruct every exact primal/dual/gap/feature equation, check 24 additional seeded designs against the independent active/sign oracle, verify exact ball containment, strict equality, stored-binary semantics, zero columns, row/column permutations, sign and scale equivariance, row replication and N normalization, general size limits, nonmutation/RNG isolation and strict CLI parity/errors. The new SVG tests all 27 visible exact cells and coordinates, both squared-bound bar geometries, full lossless report metadata, hidden dual-vector tampering and byte-exact regeneration.

The original solver, oracle, demo JSON and original figure are unchanged. Both demos and both caller-input CLI outputs are compared completely with a six-module installed wheel outside the source checkout. CI runs all 45 tests on Python 3.10-3.13, both reproducible SVG checks and all four installed-wheel parity cases.

This certifies zeros of a supplied, exact stored-input convex objective. It is not causal feature selection, predictive importance, statistical inference, a guarantee about noisy/rounded true data, adaptive solver acceleration, support completeness or a new algorithm. Rational arithmetic is educational and can be expensive; no runtime or large-data claim is made. The original float witnesses remain numerically checked rather than outward-rounded certificates. Existing history caveats and all original boundaries in [DESIGN.md](DESIGN.md) remain.
