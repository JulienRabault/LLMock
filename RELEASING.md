# Releasing LLMock

This project is designed to ship as a PyPI package, with `llmock serve` as the main user entry point.

Official references:

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [PyPI project metadata](https://docs.pypi.org/project_metadata/)
- [PyPI trusted publisher troubleshooting](https://docs.pypi.org/trusted-publishers/troubleshooting/)

## One-time setup

1. Make sure `llmock` is the PyPI project name you want to publish.
2. Sign in to PyPI and create the `llmock` project if you want to reserve the name manually first.
3. In PyPI, configure a trusted publisher for this GitHub repository.
4. Use these trusted publishing settings:

- PyPI project name: `llmock`
- GitHub Actions workflow file: `release.yml`
- Environment name: `pypi`
- Repository owner and repository name: match the GitHub repo exactly

For a first release, configure the trusted publisher on PyPI before you push the release tag.

## First publish to PyPI

1. Open the project settings for `llmock` on PyPI.
2. Add a trusted publisher that points to this repository:

- owner: `JulienRabault`
- repository: `LLMock`
- workflow: `release.yml`
- environment: `pypi`

3. Make sure the GitHub repository environment named `pypi` exists if you want to use environment protection rules later.
4. Verify that [release.yml](/C:/Users/julie/Documents/Code_space/LLMock/.github/workflows/release.yml) still matches the PyPI trusted publisher settings exactly.

If PyPI rejects the publish step, the first thing to check is that the repository name, workflow filename, branch/tag context, and environment name all match the trusted publisher registration exactly.

## Optional dry run before real PyPI

Before the real publish, you can validate the packaging flow locally:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

You can also test the wheel in a fresh venv:

```bash
python -m venv .venv-smoke
.venv-smoke/bin/pip install dist/llmock-0.1.0-py3-none-any.whl
.venv-smoke/bin/llmock --help
```

On Windows PowerShell:

```powershell
python -m venv .venv-smoke
.venv-smoke\Scripts\pip.exe install .\dist\llmock-0.1.0-py3-none-any.whl
.venv-smoke\Scripts\llmock.exe --help
```

## Release checklist

1. Update `__version__` in [llmock/__init__.py](/C:/Users/julie/Documents/Code_space/LLMock/llmock/__init__.py).
2. Review [README.md](/C:/Users/julie/Documents/Code_space/LLMock/README.md) for any install, endpoint, or behavior changes.
3. Run the local verification commands:

```bash
python -m ruff check llmock tests examples/retry_with_openai.py
pytest -q
python -m build
```

On bash or zsh, validate the built artifacts with:

```bash
python -m twine check dist/*
```

On PowerShell, use:

```powershell
python -m twine check (Get-ChildItem dist | ForEach-Object { $_.FullName })
```

4. Commit the release changes.
5. Create an annotated tag that matches the package version:

```bash
git tag -a v0.1.0 -m "LLMock v0.1.0"
git push origin main
git push origin v0.1.0
```

6. Confirm that the GitHub `Release` workflow succeeds.
7. Verify the GitHub release notes and the uploaded wheel and sdist artifacts.
8. Verify the package on PyPI and test the install path:

```bash
pipx install llmock
llmock serve --help
```

9. Sanity-check the published package as a user would:

```bash
pipx install llmock
llmock serve --response-style hello
curl http://127.0.0.1:8000/health
```

## What the workflow does

When you push a tag like `v0.1.0`, the GitHub Actions release workflow:

- installs release tooling
- runs Ruff and pytest
- builds the wheel and source distribution
- checks package metadata with Twine
- publishes the package to PyPI using trusted publishing
- creates a GitHub release with generated notes after PyPI succeeds
