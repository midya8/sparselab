# Equations, arithmetic and boundaries

## Supplied objective

For N rows, p columns, lambda > 0:

`P(beta) = ||r||^2/(2N) + lambda*sum_j |beta_j|`, `r = y-X beta`.

There is no implicit intercept, data transformation, ridge term or normalization change along the regularization path. A constant supplied column is penalized just like any other column. Scaling columns changes the statistical penalty unless lambda is changed consistently. Row replication preserves the mean-loss objective.

## Coordinate step

For column j, `q_j = sum_i X_ij^2/N`, and the recomputed partial residual is `r^(j)_i = y_i - sum_(k!=j) X_ik beta_k`. Let `rho_j = sum_i X_ij*r^(j)_i/N`. The exact one-coordinate minimizer is `soft(rho_j,lambda)/q_j`. An all-zero column is assigned coefficient zero. Partial residuals are recomputed to avoid accumulated incremental-residual drift. This is deliberately simple dense code, not a performance implementation.

A sweep visits each column exactly once in the specified permutation. The report includes the initial witness and every full-sweep objective/gap/KKT summary. No coefficient-change-only stopping rule. Both the nonnegative gap and maximum KKT residual must be <= the absolute tolerance. At a sweep limit the report retains the last coefficients with `converged=false` and `status="sweep_limit"`. A path continues warm-starting from the last iterate, including a failed fit, and its `all_converged` remains false if any fit failed. Grid points are actual fitted values, not a computed continuous LARS homotopy or exact knot sequence.

## Floating dual and KKT witnesses

The dual is `D(u)=y^T u - (N/2)||u||^2`, with `||X^T u||_infinity <= lambda`. For any feasible u, `D(u) <= P(beta*) <= P(beta)` in real arithmetic. We construct `u = s*r/N`, where `s=min(1,lambda/max_j |X_j^T r/N|)` (use 1 when the denominator is zero), then multiply s by `1-16*machine_epsilon` to stay inside the boundary. Actual computed dual correlations are checked; if they exceed lambda, scale is halved and recomputed, at most eight attempts before an explicit arithmetic error.

The residual correlation is `c_j=X_j^T r/N`. KKT requires `c_j=lambda*sign(beta_j)` on nonzero coefficients and `|c_j|<=lambda` at zero. All residuals are reported, not only the maximum.

For better cancellation behavior we also evaluate the gap as

`P-D = ||r-Nu||^2/(2N) + sum_j [lambda*|beta_j| - beta_j*(X_j^T u)]`.

Each bracket is nonnegative for a feasible u in real arithmetic. The full terms, dual vector, dual scale, dual correlations, scalar dual, and direct `P-D` subtraction are available. The interior scaling can leave a tiny positive gap even at an analytic optimum. We do not truncate the reported gap to zero. These operations use binary64 and `math.fsum`, not interval arithmetic or outward rounding. The computed feasibility test is not a rigorous real-arithmetic certificate, and the reported gap does not bound coefficient distance under rank deficiency. Tiny feature perturbations, extreme scaling, conditioning and sweep budgets can matter. Do not use this as a safety or statistical guarantee.

## Independent exact active/sign enumeration

For p<=5 and N<=32, convert validated float inputs using `Fraction(float)`. Thus 0.1 means the exact stored binary64 value, not 1/10. No float inverses, tolerance-based sign checks or descent updates are reused by the oracle.

For every active subset A and sign vector s, solve

`(X_A^T X_A/N) beta_A = X_A^T y/N - lambda*s`

using rational Gauss-Jordan elimination with row swaps. Require every active coefficient to have its specified strict sign, then check **all** KKT conditions exactly. Empty support is included. Exactly singular active systems are counted and skipped. The returned witnesses are nonsingular-support optima, not a listing of every minimizer. LASSO with lambda>0 has an optimal independent-column support: a dependent nonzero active support has a coefficient null direction; at optimum its sign-region L1 directional derivative must be zero, so one may move until a coefficient hits zero without changing prediction or objective, then repeat. This explains why skipping singular systems does not need an invented ridge term. The enumeration is exponential and educational, not a replacement for large solvers.

Strict convexity in the fitted prediction implies all optima have identical predictions: if two differed, the squared-loss midpoint would strictly improve while L1 is convex. Coefficients can still differ along the design nullspace. In the duplicate fixture the exact enumerator returns `[7/4,0,-1/2]` and `[0,7/4,-1/2]`, both with objective `5/8` and predictions `[5/4,9/4,-9/4,-5/4]`. Every convex mixture is optimal. These statements concern a supplied optimization problem, not data-generating causality or feature relevance.

## Controls and report

Orthogonal columns are the three nonconstant columns of a four-row Hadamard design. `y = 2*a - 3/4*b + 1/8*c`; X^T X/N is identity. Six dyadic penalties compare closed-form soft thresholds, independent exact enumerations and coordinate results. At lambda_max=2 zero is optimal. At lambda=1/16 coefficients are `[31/16,-11/16,1/16]`.

Duplicate design columns are `[a,a,b]`, response `2*a-3/4*b`, lambda=1/4. Both runs start from zero; only visit order differs. No randomized tie-breaking or perturbed data. Midpoint is audited directly with the same primal/dual equations.

Correlated columns are `[a,a+b/4]`, response `2*a+3*b/4`, lambda=1/8. Compare one sweep with the same solver allowed to converge. One sweep yields `[15/8,3/17]`, primal approximately 0.506893382 and gap 0.330882353. After 18 sweeps, coefficients are `[0,33/17]`, exact optimum `19/68`. The first run is explicitly a truncated diagnostic, not an alternative optimization baseline. Full fit histories are in the demo JSON and compressed lossless SVG metadata.

## API and resource limits

- `fit(X,y,penalty,tolerance=1e-10,max_sweeps=20000,initial=None,order=None)`.
- `path(X,y,penalties,tolerance=1e-10,max_sweeps=20000)`: 1..32 strictly descending, distinct positive penalties, warm starts. No inference that supports are nested or that all requested fits converge.
- `witness(X,y,coefficients,penalty)`: evaluate arbitrary supplied coefficients without fitting.
- `enumerate_optima(X,y,penalty)`: <=32 rows, <=5 columns, up to 3^5 exact sign patterns; rational strings plus float-independent exact acceptance checks.
- General X: 1..128 rows, 1..12 columns; finite numeric entries and y in [-100,100], booleans rejected. Each nonzero column must have mean squared norm >=1e-8; zero columns remain allowed.
- penalty in [1e-8,10000], tolerance in [1e-12,1e-5], max_sweeps integer in [1,50000]. Initial/audited coefficients in [-1e6,1e6]. Column order is an exact integer permutation, no booleans. Inputs copied, global RNG untouched.
- JSON CLI: exactly integer schema_version=1 plus X,y,penalties; duplicate keys, unknown/missing fields, nonfinite values, invalid UTF-8 and files >100000 bytes rejected with exit code 2. CLI path reports sweep-limit failures in JSON rather than treating them as input errors. Output may be much larger than input because every sweep is retained. Python API permits a lower sweep budget than the CLI, which uses defaults.

## Provenance and non-goals

Standard cyclic coordinate descent/soft thresholding and warm regularization paths follow [Friedman, Hastie and Tibshirani, Section 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929880/). Standard KKT and dual geometry are covered in [Ryan Tibshirani's coordinate descent notes](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/coord-desc.pdf). Normalization in this implementation is explicitly the mean squared loss. Original code implements those standard equations; no algorithmic novelty is claimed.

No external dataset, runtime dependency, network call, speed benchmark, estimator comparison, cross-validation, statistical interval, feature-causality claim, post-selection inference or production certification. The figure and comparisons were developed alongside the implementation and are descriptive mathematical controls, not held-out empirical evaluation.
