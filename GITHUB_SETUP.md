# Put this package on GitHub

## 1. Create the repository

Create an empty public repository named `landfeedback` in your GitHub account.
Do not initialize it with another README, license, or `.gitignore`, because all
three are already included here.

## 2. Push this folder

Open a terminal inside the extracted `landfeedback` folder and run:

```bash
git init
git add .
git commit -m "Initial public release of landfeedback 0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/landfeedback.git
git push -u origin main
```

Replace `YOUR-GITHUB-USERNAME` with your actual GitHub username.

## 3. Confirm automated checks

Open the repository's **Actions** page. The `CI` workflow should test Python
3.10-3.12 on Linux, Windows, and macOS and should build both package
distributions.

If GitHub Actions is disabled for the repository, enable it under
**Settings > Actions > General**.

## 4. Add repository links

After the repository URL is known, add this section to `pyproject.toml`:

```toml
[project.urls]
Documentation = "https://YOUR-GITHUB-USERNAME.github.io/landfeedback/"
Issues = "https://github.com/YOUR-GITHUB-USERNAME/landfeedback/issues"
Repository = "https://github.com/YOUR-GITHUB-USERNAME/landfeedback"
```

Commit and push the change.

## 5. Create the first GitHub release

After CI passes:

```bash
git tag -a v0.1.0 -m "landfeedback 0.1.0"
git push origin v0.1.0
```

On GitHub, open **Releases**, draft a release from tag `v0.1.0`, and use the
Version 0.1.0 section of `CHANGELOG.md` as the release notes.

## 6. Before publishing to PyPI

Confirm that the project name is still available, replace any temporary contact
information, run the checks below in a fresh environment, and test the built
wheel:

```bash
python -m pip install -e ".[dev,docs]"
ruff check .
pytest --cov=landfeedback --cov-report=term-missing
python -m build
python -m twine check dist/*
```

Do not claim exact reproduction of the full Brubaker-Entekhabi physical model
until the two 1995 companion formulations have been implemented and validated.

