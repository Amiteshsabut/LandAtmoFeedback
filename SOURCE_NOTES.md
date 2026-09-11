# Source and validation notes

This validation build was prepared against:

Brubaker, K. L., & Entekhabi, D. (1996). *Analysis of feedback mechanisms in
land-atmosphere interaction*. Water Resources Research, 32(5), 1343-1357.
https://doi.org/10.1029/96WR00005

## Directly represented from the 1996 article

- Table 2 principal parameter values.
- Table 3 equilibrium, stationary mean, and stationary standard deviations.
- Table 4 deterministic soil-moisture decomposition.
- Table 5 deterministic soil-temperature decomposition.
- Tables 6-7 fifth/95th-percentile conditional contributions and nonlinear comparison values.
- Tables 8-9 stochastic perturbation-forcing coefficients.
- Eq. 19-20 coefficient convention: derivative times predictor standard deviation.
- Eq. 32 stochastic susceptibility linearization logic.
- Treatment of the land-air temperature difference as a derived reporting channel rather than an independent prognostic state.

## Deliberate boundary

The 1996 paper states that the model was developed in companion 1995 papers and refers readers there for further details. This build therefore does not invent missing parameters or claim an end-to-end reconstruction of the complete four-state physical model from the 1996 article alone.

Instead, the package has two validation layers:

1. independent numerical verification using analytic nonlinear drift and diffusion functions with exact derivatives; and
2. published-value regression/arithmetic validation against the 1996 tables.

That distinction should be retained in publications and documentation.
