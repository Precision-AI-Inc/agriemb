# Contributing

Thank you for your interest in contributing to this Precision AI project.

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd agriemb

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

`pip install -e ".[dev]"` installs all dependencies including pre-commit. `pre-commit install` registers the hooks so they run automatically on every commit.

## Making changes

1. Create a branch from `main`.
2. Make your changes.
3. Commit. Pre-commit hooks run automatically and will block the commit if any check fails.

If a hook auto-fixes files (ruff lint/format), stage the changes and commit again.

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (>1800 KB), trailing whitespace, merge conflicts, private keys, debug statements, BOM removal |
| `ruff-check` | Linting and import sorting (auto-fix); includes `ANN` rules that enforce PEP 484 annotations |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking (pylance engine) — config in `[tool.pyright]` in `pyproject.toml` |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

## Tests

Tests live in `tests/` and are pure and offline - they need no dataset download, no API key, and no network:

```bash
pytest
```

## Continuous integration

Pushes to `main` and pull requests run `.github/workflows/ci.yml`, which dispatches to three jobs:

| Job | What it runs |
|---|---|
| `pre-commit` | All hooks above, across all files |
| `test` | `pytest`, preceded by an import smoke-test of `precisionai.agriemb` from the repo root |
| `docs` | A strict Sphinx build (`make -C docs html` passes `-W --keep-going`), then a check that the generated Module Reference is not empty |

The release workflow runs the same three jobs before it builds or publishes anything, so a tag pushed from a red commit cannot produce a GitHub Release.

## Code style

- **Formatter / linter:** ruff (`line-length = 120`, Python 3.10 target).
- **Type hints:** all functions (public and private) must have complete PEP 484 type annotations. Enforced at lint time by ruff (`ANN` rules) and statically by pyright. `Any` is allowed for genuinely dynamic types; explicit `Any` is preferred over `object` when methods will be called on a value.
- **Docstrings:** NumPy style for all public functions, classes, and modules.
- **Comments:** only where the _why_ is non-obvious. No inline narration of what the code does.

See [CLAUDE.md](CLAUDE.md) for the full standard.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Write a clear description of what changed and why.
- All CI checks - pre-commit hooks, tests, and the docs build - must pass before requesting review.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
