# LandAtmoFeedback

A Python toolkit for local linearization, restoring/reinforcing feedback
diagnostics, process attribution, conditional dry/moist composites, stochastic
susceptibility, and scientific validation in coupled land-atmosphere systems.

## What changed in this validated build

This `0.2.0` validation build explicitly aligns the numerical definitions with
Brubaker & Entekhabi (1996), *Water Resources Research*, 32(5), 1343-1357.

The central deterministic coefficient is

\[
A_{ij} =
\left.\frac{\partial G_i}{\partial x_j}\right|_{x^*}\sigma_j,
\]

where `sigma_j` is the stationary standard deviation of predictor state `j`.
This is the paper's one-sigma scaled coefficient convention.  It has units of
the **target tendency**; it is not a dimensionless standardized Jacobian.

For a standardized disequilibrium

\[
\delta_j=(x_j-x_j^*)/\sigma_j,
\]

the linearized pathway contribution is

\[
A_{ij}\delta_j.
\]

The package classifies a contribution as restoring when it opposes the target
anomaly and reinforcing when it has the same sign.

For stochastic forcing, the package follows the analogous expansion

\[
g_i(x)\approx g_i(x^*)+\sum_j\Lambda_{ij}\delta_j,
\qquad
\Lambda_{ij}=
\left.\frac{\partial g_i}{\partial x_j}\right|_{x^*}\sigma_j.
\]

## Installation

```bash
python -m pip install -e .
```

## Run the full validation suite

```bash
landfeedback validate
```

or:

```python
from landfeedback import full_validation_report

report = full_validation_report()
print(report.summary())
report.assert_valid()
```

The validation combines:

- exact analytic Jacobian verification for a nonlinear coupled system;
- central/forward/complex-step derivative cross-checks;
- derivative step-size convergence;
- process-budget closure;
- one-sigma coefficient scaling;
- pathway contribution closure;
- local Taylor-error behavior;
- stochastic/diffusion linearization;
- published-value regression checks against Brubaker & Entekhabi (1996)
  Tables 3-9.

## Brubaker & Entekhabi (1996) workflow

The paper does **not** compute a different Jacobian for dry and moist tails.
It linearizes at the equilibrium and then combines those equilibrium
coefficients with physically consistent conditional disequilibria from the
joint stochastic distribution at the dry (<=5th percentile) and moist
(>=95th percentile) soil-moisture tails. `conditional_composites` follows that
logic.

The paper also reports `delta_temperature = soil_temperature - air_temperature`
as an additional reporting channel even though the prognostic system contains
four states. `ReportingDecomposition` supports such derived reporting channels
without falsely treating them as independent model states.

## Important validation boundary

The package contains published numerical fixtures from the 1996 analysis.
Those fixtures validate conventions and reported arithmetic. They are **not**
presented as a complete independent reconstruction of the original four-state
physical model, because the 1996 article itself refers to the two 1995
companion papers for additional model construction and parameterization
details.

See `docs/validation.md` and `examples/brubaker1996_validation.py`.
