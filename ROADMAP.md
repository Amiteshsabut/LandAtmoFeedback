# Roadmap

## 0.1 series: local model diagnostics

- Stabilize the generic model, equilibrium, derivative, feedback, process,
  conditional-composite, stochastic, plotting, and benchmark APIs.
- Add independent user examples and uncertainty guidance.

## 0.2: uncertainty and gridded workflows

- Bootstrap intervals for feedback coefficients and conditional composites.
- `xarray` output objects and Dask-compatible gridded execution.
- NetCDF and Zarr export conventions.
- Seasonal-cycle and lag-sensitivity utilities for reanalysis data.

## 0.3: empirical local dynamics

- Regularized local-linear tendency estimation from time series.
- Blocked cross-validation and residual diagnostics.
- Collinearity, identifiability, and effective-sample-size warnings.
- Clear separation between dynamical association and causal attribution.

## 1.0: independently validated scientific release

- Implement the complete Brubaker-Entekhabi 1995-1996 physical model.
- Reproduce equilibrium, stationary statistics, and published feedback tables
  within documented tolerances.
- Validate against an independent land-surface model or observational case.
- Archive a release with a DOI and submit a peer-reviewed software paper.

