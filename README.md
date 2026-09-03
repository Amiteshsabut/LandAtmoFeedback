
<h1 align="center">LandAtmoFeedback</h1>

<p align="center">
  A Python toolkit for diagnosing coupled land–atmosphere feedbacks,<br>
  stability, process attribution, conditional composites, and stochastic sensitivity.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/Amiteshsabut/LandAtmoFeedback/graphs/contributors"><img src="https://img.shields.io/github/contributors/Amiteshsabut/LandAtmoFeedback.svg" alt="Contributors"></a>
  <a href="https://github.com/Amiteshsabut/LandAtmoFeedback/issues"><img src="https://img.shields.io/github/issues/Amiteshsabut/LandAtmoFeedback.svg" alt="Issues"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Amiteshsabut/LandAtmoFeedback.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#key-capabilities">Capabilities</a> |
  <a href="#scientific-framework">Framework</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#brubaker--entekhabi-1996-benchmark">Benchmark</a> |
  <a href="#documentation">Documentation</a> |
  <a href="#citation">Citation</a> |
  <a href="#contact">Contact</a>
</p>

## Overview

**LandAtmoFeedback** provides reusable numerical tools for studying feedbacks in coupled land–atmosphere models. It evaluates a user-defined tendency function around an equilibrium, constructs its local Jacobian, diagnoses stability and relaxation modes, attributes feedbacks to individual physical processes, and compares behavior across conditional or stochastic states.

The GitHub project is named **LandAtmoFeedback**; the installable Python distribution and import package are named **`landfeedback`**.

The toolkit is designed for transparent, reproducible research rather than for one fixed model. Users supply their own state variables, parameters, governing tendency function, and when needed process-level tendencies.

## Key Capabilities

| Capability | What it provides |
| --- | --- |
| Equilibrium diagnostics | Solve for, verify, and report steady states of coupled models |
| Numerical linearization | Central, forward, or complex-step Jacobian estimation |
| Feedback analysis | Raw and scaled feedback matrices with restoring/reinforcing classification |
| Process attribution | Process-level Jacobians and closure checks against the total tendency |
| Stability analysis | Eigenvalues, eigenvectors, relaxation timescales, and dominant modes |
| Conditional analysis | Lower-tail, upper-tail, and asymmetry composites for dry/moist states |
| Stochastic sensitivity | Diffusion sensitivities and Euler–Maruyama simulations |
| Benchmarking | Curated tables from Brubaker & Entekhabi (1996) as regression fixtures |
| Reproducible workflows | Examples, tests, documentation, command-line tools, and GitHub Actions |

## Scientific Framework

For a state vector $`\mathbf{x}`$, parameters $`\boldsymbol{\theta}`$, and model tendency $`\mathbf{F}`$, the package represents the coupled system as

```math
\frac{d\mathbf{x}}{dt} = \mathbf{F}(\mathbf{x},\boldsymbol{\theta}).
```

At an equilibrium $`\mathbf{x}^{*}`$, where $`\mathbf{F}(\mathbf{x}^{*},\boldsymbol{\theta}) \approx \mathbf{0}`$, small perturbations evolve according to

```math
\frac{d\,\delta\mathbf{x}}{dt} \approx \mathbf{J}\,\delta\mathbf{x},
\qquad
J_{ij} = \left.\frac{\partial F_i}{\partial x_j}\right|_{\mathbf{x}=\mathbf{x}^{*}}.
```

The Jacobian $`\mathbf{J}`$ describes local interactions among state variables. Its eigenvalues diagnose linear stability and relaxation rates, while process-specific Jacobians reveal how individual modeled processes contribute to the total response.

These diagnostics are **local and model-conditioned**. They should not automatically be interpreted as causal relationships in observations.

## Installation

### Install from GitHub

```bash
git clone https://github.com/Amiteshsabut/LandAtmoFeedback.git
cd LandAtmoFeedback
python -m pip install -e ".[plot]"
```

### Development installation

```bash
python -m pip install -e ".[dev,docs]"
```

After a future PyPI release, the package will be installable with:

```bash
pip install landfeedback
```

## Quick Start

Define a model tendency, create a `CoupledModel`, and analyze an equilibrium:

```python
import numpy as np

from landfeedback import CoupledModel, FeedbackAnalyzer


def tendency(state):
    soil_moisture, temperature = state
    return np.array([
        -0.40 * soil_moisture - 0.10 * temperature,
         0.25 * soil_moisture - 0.60 * temperature,
    ])


model = CoupledModel(
    state_names=("soil_moisture", "temperature"),
    tendency=tendency,
)

analyzer = FeedbackAnalyzer(model)
result = analyzer.analyze(
    equilibrium=np.array([0.0, 0.0]),
    scales=np.array([1.0, 1.0]),
)

print(result.jacobian)
print(result.locally_stable)
print(result.relaxation_modes())
```

The returned `FeedbackResult` contains the equilibrium, Jacobian, scaled feedback matrix, eigensystem, stability classification, and relaxation diagnostics.

### Attribute feedbacks to physical processes

Provide separate tendency functions for named processes:

```python
def radiation(state):
    soil_moisture, temperature = state
    return np.array([0.0, -0.35 * temperature])


def hydrology(state):
    soil_moisture, temperature = state
    return np.array([
        -0.40 * soil_moisture - 0.10 * temperature,
         0.25 * soil_moisture - 0.25 * temperature,
    ])


model = CoupledModel(
    state_names=("soil_moisture", "temperature"),
    tendency=tendency,
    processes={"radiation": radiation, "hydrology": hydrology},
)

result = FeedbackAnalyzer(model).analyze(
    equilibrium=np.array([0.0, 0.0]),
    scales=np.array([1.0, 1.0]),
)
print(result.process_jacobians)
print(result.process_closure())
```

### Compare dry and moist conditional states

```python
from landfeedback import conditional_composites

composite = conditional_composites(
    result=result,
    states=sampled_states,
    conditioning="soil_moisture",
    lower_quantile=0.20,
    upper_quantile=0.80,
)

print(composite.lower_mean_state)
print(composite.upper_mean_state)
print(composite.asymmetry_index())
```

### Run a stochastic experiment

```python
from landfeedback import CoupledModel, simulate_sde

stochastic_model = CoupledModel(
    state_names=("soil_moisture", "temperature"),
    tendency=tendency,
    diffusion=lambda state: np.array([0.03, 0.02]),
)

trajectory = simulate_sde(
    model=stochastic_model,
    x0=np.array([0.0, 0.0]),
    dt=0.01,
    n_steps=10_000,
    seed=42,
)
```

See [`examples/`](examples/) for complete, executable workflows.

## Brubaker & Entekhabi (1996) Benchmark

The package includes machine-readable reference values transcribed from Tables 2–9 of:

> Brubaker, K. L., & Entekhabi, D. (1996). Analysis of feedback mechanisms in land–atmosphere interaction. *Water Resources Research, 32*(5), 1343–1357. [https://doi.org/10.1029/96WR00005](https://doi.org/10.1029/96WR00005)

Inspect a table from the command line:

```bash
landfeedback benchmark states
landfeedback benchmark reporting-matrix
landfeedback benchmark moisture-stochastic --csv
```

Or access it in Python:

```python
from landfeedback.benchmarks import brubaker1996

print(brubaker1996.state_statistics())
print(brubaker1996.published_reporting_matrix())
```

### Benchmark scope

The bundled published values are **regression fixtures**, not an exact reproduction of the complete physical model. A faithful end-to-end reconstruction also requires equations and parameterizations documented in the companion 1995 studies cited by Brubaker & Entekhabi. In addition, derived quantities such as \(\Delta T\) should not be introduced as independent state variables unless the chosen model formulation explicitly treats them that way.

## Repository Structure

```text
LandAtmoFeedback/
├── src/landfeedback/       # Installable Python package
├── tests/                  # Unit and regression tests
├── examples/               # Runnable scientific examples
├── docs/                   # User guide, theory, API, and banner assets
├── .github/workflows/      # Continuous-integration configuration
├── pyproject.toml          # Build metadata and dependencies
├── CITATION.cff            # Citation metadata
├── LICENSE                 # MIT License
└── README.md               # Project overview
```

## Validation

Run the complete test and quality suite with:

```bash
python -m pytest
python -m ruff check .
python -m mypy src/landfeedback
```

GitHub Actions runs the automated checks across the supported Python versions.

## Documentation

- [Getting started](docs/quickstart.md)
- [Scientific theory](docs/theory.md)
- [API reference](docs/api.md)
- [Benchmark provenance](docs/benchmark.md)
- [Contributing guide](CONTRIBUTING.md)
- [Project roadmap](ROADMAP.md)
- [Security policy](SECURITY.md)

## Citation

If you use LandAtmoFeedback in research, cite the software version you used. GitHub can generate citation text from [`CITATION.cff`](CITATION.cff) through the repository's **Cite this repository** menu.

If you use the included Brubaker–Entekhabi benchmark data, also cite the original article:

```bibtex
@article{brubaker1996feedback,
  author  = {Brubaker, Kaye L. and Entekhabi, Dara},
  title   = {Analysis of Feedback Mechanisms in Land--Atmosphere Interaction},
  journal = {Water Resources Research},
  year    = {1996},
  volume  = {32},
  number  = {5},
  pages   = {1343--1357},
  doi     = {10.1029/96WR00005}
}
```

## Contributing

Contributions, bug reports, benchmark corrections, and documentation improvements are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and open an issue before proposing a substantial change.

## Contact

- Maintainer: [Amitesh Sabut](https://github.com/Amiteshsabut)
- Questions and bug reports: [GitHub Issues](https://github.com/Amiteshsabut/LandAtmoFeedback/issues)

## License

LandAtmoFeedback is distributed under the [MIT License](LICENSE).

## Disclaimer

LandAtmoFeedback is academic research software. Its results depend on the equations, parameters, equilibrium, numerical settings, and data supplied by the user. The package is not an operational weather or climate prediction system and should not be used by itself for risk assessment, safety-critical decisions, or policy decisions. Interpret results alongside physical reasoning, uncertainty analysis, independent validation, and expert judgment.
