# SparseLab

A dependency-free LASSO reference that makes optimality and coefficient ambiguity visible. Fit small supplied regressions, inspect every sweep, and compare floating-point primal/dual/KKT witnesses with an independent exact-rational active/sign enumerator.

**A unique fitted prediction does not imply a unique selected feature.** In the duplicate-column control, forward sweeps select `[1.75, 0, -0.5]`, reverse sweeps select `[0, 1.75, -0.5]`, and their midpoint `[0.875, 0.875, -0.5]` is equally optimal. All have objective `5/8`, identical predictions and zero KKT residual. This is optimization geometry, not causal feature importance.

![LASSO path, duplicate-coefficient ambiguity and early-stopping audits](assets/sparsity-audit.svg)

## Run offline

Python 3.10+, standard library at runtime. No credentials, downloads, datasets or paid APIs.

```bash
python -m sparselab demo
python -m sparselab path examples/input.json
python -m unittest discover -s tests -v
python examples/render_figure.py --check
python -m pip wheel --no-deps . -w dist
```

The demo asserts both a successful fit and a one-sweep nonconvergence control. [Full demo JSON](examples/demo.json), [caller input](examples/input.json) and [equations and limits](DESIGN.md) are committed. CLI inputs are capped at 100,000 bytes; reports may be larger because they include the full sweep history.

```python
from sparselab import fit, path, witness, enumerate_optima

result = fit([[1, 1], [1, -1]], [2, 0], penalty=0.25)
assert result['converged']
print(result['coefficients'], result['primal'], result['dual'], result['gap'])
print(result['correlations'], result['dual_vector'], result['max_kkt'])

# Strictly descending positive penalties; each fit starts from the previous fit.
report = path([[1, 1], [1, -1]], [2, 0], [1, 0.5, 0.25])
exact = enumerate_optima([[1, 1], [1, -1]], [2, 0], 0.25)
```

## What is actually computed

- Minimize `||y-X beta||^2/(2N) + lambda*||beta||_1`. No hidden intercept, centering or standardization. Every supplied column, including an explicit constant column, is penalized.
- Cyclic soft-threshold updates with recomputed partial residuals, configurable column order and initialization. Zero columns are handled explicitly.
- At every full sweep: primal objective, feasible scaled-residual dual vector, all dual correlations, stable decomposed gap, direct primal-minus-dual difference, every KKT residual and support.
- Stop only when **both absolute gap and maximum KKT residual are at most `tolerance`**. Otherwise return `status="sweep_limit"`, not a fabricated success. `path.all_converged` propagates failures.
- A separate bounded `Fraction` oracle enumerates `3^p` inactive/positive/negative patterns, solves nonsingular active normal equations with exact rational elimination, and accepts only exact KKT witnesses. Fractions represent the validated binary64 inputs exactly, not decimal idealizations. Singular patterns are counted and skipped, not ridge-regularized.

## Three developed controls

| Control | Observed result | Interpretation |
|---|---|---|
| Orthogonal Hadamard columns | Six penalties `[2,1,1/2,1/4,1/8,1/16]`; coefficients exactly follow soft-thresholding of `[2,-3/4,1/8]` | Independent closed forms, not a real-data benchmark |
| Duplicate columns | Two exact sparse witnesses, same predictions and exact objective `5/8`; their continuum of convex mixtures is optimal | Feature identity/support is not identified by the fit |
| Correlated columns | One sweep: gap `0.330882353`, max KKT `0.176470588`; converged at 18 sweeps with objective `19/68` | A sweep limit is not an optimality check; exact oracle agrees with the final fit |

The one-sweep run uses the **same objective, data, initialization and order** as the converged run. It is an explicitly truncated negative control, not a competitive baseline. No speed or optimizer-superiority claim.

## Verification

28 tests: 25 behavior/CLI tests plus 3 figure tests. Analytic scalar/orthogonal controls, ten seeded exact-rational comparisons, duplicate solutions and convex mixtures, primal/dual equations and weak duality, history monotonicity, warm/cold agreement, response/feature signs, row/column permutation, row replication, scale/penalty equivalence, zero columns, honest sweep limits, strict malformed inputs, nonmutation and unchanged global RNG state.

The SVG checks cover all 18 coefficient points/polyline geometry, all 23 summary cells and their coordinates, full compressed JSON metadata, hidden intermediate-history tampering, and byte-exact regeneration. CI runs Python 3.10-3.13. A five-module wheel must reproduce both complete CLI JSON outputs outside the source checkout.

## Scope and attribution

This is a small dense numerical reference, not a scalable sparse solver. Fits accept at most 128 rows and 12 columns; the exact oracle accepts at most 32 rows and 5 columns. Positive penalties only. Supplied entries are bounded and nonzero columns need mean squared norm at least `1e-8`. See [DESIGN.md](DESIGN.md) for all limits and arithmetic qualifications.

Floating-point dual/KKT witnesses are **not directed-rounding certificates**. Positive ridge is not silently added to remove nonuniqueness. Coefficients depend on feature units and may be nonunique. No cross-validation, held-out prediction claim, feature causality, statistical coverage, post-selection inference, logistic model or new optimization algorithm.

Standard methods: Friedman, Hastie and Tibshirani, [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929880/), especially Section 2; Ryan Tibshirani's [Coordinate Descent notes](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/coord-desc.pdf) for soft-threshold/KKT and LASSO dual geometry. SparseLab's implementation, exact witnesses, controls and report format are original code applying those methods. MIT license.
