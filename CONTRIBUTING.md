# Contributing to LLMock

Thanks for contributing.

LLMock is intentionally narrow in scope: it is a local mock server for testing LLM client resilience, retries, fallbacks, and integration behavior. Contributions are most useful when they improve fidelity, reliability, or adoption without turning the project into a full provider emulator.

## Good contribution targets

- fix mismatches between runtime behavior and provider docs
- add tests for provider-specific request and error shapes
- improve batch endpoint coverage
- improve docs, examples, and onboarding
- add deterministic testing capabilities
- tighten packaging, release automation, and CI

## Changes that need extra care

- large new provider surfaces
- features that increase maintenance cost a lot
- behavior that diverges from official provider APIs without being documented
- breaking CLI, config, or response-shape changes

If a change is large or changes public behavior, open an issue first.

## Local setup

```bash
pip install -e ".[dev]"
```

Run checks before opening a pull request:

```bash
python -m ruff check llmock tests examples/retry_with_openai.py
pytest -q
python -m build
```

## Pull request expectations

- keep changes scoped and explain the user-facing impact clearly
- add or update tests for behavioral changes
- update [README.md](/C:/Users/julie/Documents/Code_space/LLMock/README.md) when install, config, or API behavior changes
- preserve provider-specific response and error shapes where possible
- avoid unrelated refactors in the same pull request

## Design principles

- local-first and cheap to run
- explicit over magical
- useful for resilience testing before it is exhaustive
- predictable defaults, configurable failure modes
