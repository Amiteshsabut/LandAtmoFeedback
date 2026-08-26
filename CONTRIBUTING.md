# Contributing

Contributions are welcome, especially independently checked implementations of
land-surface process terms, additional published benchmarks, uncertainty
methods, and gridded `xarray` workflows.

## Development setup

```bash
git clone <repository-url>
cd landfeedback
python -m venv .venv
```

Activate the environment and install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Before opening a pull request

```bash
ruff check .
pytest --cov=landfeedback --cov-report=term-missing
python -m build
python -m twine check dist/*
```

New scientific functionality must include:

- a documented mathematical definition;
- units and coordinate conventions;
- a synthetic test with a known answer;
- a literature citation when reproducing a published method;
- uncertainty or numerical-tolerance discussion when appropriate.

Do not describe empirical feedback estimates as causal effects without a
design that supports causal identification.

