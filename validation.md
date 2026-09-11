# Scientific validation

## 1. What the 1996 paper actually does

Brubaker & Entekhabi (1996) analyze a four-state stochastic land-atmosphere
system

`dx = G(x) dt + g(x) dW`

with soil moisture, mixed-layer humidity, soil temperature, and mixed-layer
potential temperature as prognostic states.

Their analysis has three essential deterministic steps:

1. linearize `G` at a stable equilibrium `x*`;
2. scale each predictor derivative by that predictor's stationary standard
   deviation, `A_ij = (dG_i/dx_j)|x* sigma_j`;
3. multiply the coefficients by physically consistent standardized
   disequilibria drawn from conditional tails of the joint stochastic state
   distribution.

The dry state is the <=5th percentile of soil moisture and the moist state is
the >=95th percentile.

They then compare the summed linearized tendency with the conditional mean of
the original nonlinear tendency.

The stochastic forcing function is treated analogously:

`g_i(x) ~= g_i(x*) + sum_j Lambda_ij delta_j`.

## 2. What this package validates

`full_validation_report()` checks numerical correctness independently using a
nonlinear system with exact analytic derivatives. It verifies:

- deterministic Jacobian;
- complex-step cross-check;
- diffusion Jacobian;
- deterministic process closure;
- paper-style one-sigma scaling;
- contribution-sum identity;
- local Taylor-error contraction;
- diffusion linearization;
- standardized-coordinate eigenvalue invariance.

It also validates the arithmetic and stored published values from the 1996
paper's Tables 3-9.

## 3. What is not claimed

The 1996 paper explicitly points to companion 1995 studies for fuller model
construction and stochastic parameterization. Therefore the bundled 1996
tables are regression/benchmark fixtures, not a claim that the entire original
physical model has been independently reconstructed from the 1996 article
alone.

## 4. Why `scaled_matrix = J * sigma_j` is correct here

This is exactly the coefficient convention used by the paper.  A separate
`standardized_dynamics_matrix = D^-1 J D` is provided only as an additional
dimensionless-coordinate diagnostic.  It should not replace the paper-style
coefficient when reproducing the 1996 analysis.

## 5. Derived temperature-difference channel

The paper rewrites part of the temperature dependence using
`Delta = Tg - theta_m`.  Delta is not an independent fifth state.
`ReportingDecomposition` exists so this reporting convention can be reproduced
without changing the model-state dimension.
