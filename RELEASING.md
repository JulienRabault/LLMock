# Releasing LLMock

This project is designed to ship as a PyPI package, with `llmock serve` as the main user entry point.

## One-time setup

1. Make sure `llmock` is the PyPI project name you want to publish.
2. In PyPI, configure a trusted publisher for this GitHub repository.
3. Use these trusted publishing settings:

- PyPI project name: `llmock`
- GitHub Actions workflow file: `release.yml`
- Environment name: `pypi`
- Repository owner and repository name: match the GitHub repo exactly

For a first release, configure the trusted publisher on PyPI before you push the release tag.

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

## What the workflow does

When you push a tag like `v0.1.0`, the GitHub Actions release workflow:

- installs release tooling
- runs Ruff and pytest
- builds the wheel and source distribution
- checks package metadata with Twine
- publishes the package to PyPI using trusted publishing
- creates a GitHub release with generated notes after PyPI succeeds
