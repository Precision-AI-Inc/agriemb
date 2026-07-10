# Precision AI — Python Project Standards

Apply these standards when writing, reviewing, or refactoring code in any PAI Python project.

---

## Using this template

When starting a new project from this template, complete these steps in order.

**1. Global find-and-replace** (in the order shown — `myproject` last to avoid partial matches):

| Find | Replace with | Notes |
|---|---|---|
| `agri-template` | your GitHub repository name | badge/logo URLs in `README.md`, comment in `.github/workflows/ci.yml`, issue template links |
| `pai-myproject` | your distribution name (kebab-case, e.g. `pai-ag-emb`) | `pyproject.toml` `[project]` name and `[project.scripts]`, PyPI badge URLs |
| `PAI MyProject` | your human-readable project name | `pyproject.toml` description, `docs/conf.py`, `docs/index.rst`, `docs/modules.rst` |
| `myproject` | your project namespace (snake_case, e.g. `ag_emb`) | all Python source files, all `docs/` files, `pyproject.toml` |

The `precisionai` top-level namespace package is **fixed** across all PAI Python projects — never rename it. Your project lives at `precisionai/<namespace>/`.

**2. Update `pyproject.toml`:**
- `description` — one-line description of what this project does
- `dependencies` — remove FastAPI/uvicorn if not building an API; add domain-specific deps
- `keywords`, `classifiers`, `[project.urls]` — adjust for your domain and repository

**3. Replace the hello-world scaffold** in `precisionai/<namespace>/` with real domain logic, following the layer rules documented in [Project layout](#project-layout) below.

**4. Update `docs/`** — `conf.py` project name and header, `index.rst` and `modules.rst` autodoc references.

**5. Review `LICENSE.md`** — update the copyright year if needed. The license is Apache 2.0.

**6. Seed `CHANGELOG.md`** with your first release notes under `[Unreleased]` — see [Releasing](#releasing).

**7. Remove this section** from `CLAUDE.md` once setup is complete.

---

## Environment

- Python 3.10+, managed with `python -m venv .venv` (never conda)
- Install: `pip install -e ".[dev]"` → installs all deps including pre-commit
- Register hooks once: `pre-commit install`

---

## Project layout

```
precisionai/<namespace>/
  api/
    routes/      # FastAPI route handlers (thin — delegate to services)
    config.py    # env-var config only
    app.py       # FastAPI app factory
  schemas/       # Pydantic v2 request/response models
  services/      # Business logic (orchestrates metrics, handles I/O)
  metrics/       # Pure computation modules
    __init__.py  # re-exports only — no logic
docs/            # Sphinx (HTML + LaTeX/PDF)
tests/           # mirrors package structure
examples/        # standalone runnable scripts
```

`precisionai` is a fixed, namespace-only top-level package (no logic, just an SPDX header) shared by every PAI Python project — it is what makes `precisionai.<namespace>.*` imports consistent across the org. Multi-product repositories (e.g. one repo serving several related tools) may add another level, `precisionai/<product>/<namespace>/...`.

---

## pyproject.toml — canonical config

```toml
[project]
dynamic = ["version"]   # version comes from the git tag via setuptools-scm, never hardcoded

[tool.setuptools_scm]
version_scheme = "no-guess-dev"
local_scheme = "no-local-version"
fallback_version = "0.0.0"

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E","W","F","I","UP","B","SIM","N","C90","D","PT","RUF","PL","ANN","PERF","S"]
ignore = [
    "E501",    # enforced by ruff-format
    "B008",    # FastAPI default-arg pattern
    "SIM108",  # ternary readability
    "D100","D104",          # module/package docstrings optional
    "D203","D213",          # pydocstyle conflicts — always ignore these two
    "PLR0913","PLR2004",    # arg count + magic values common in metrics/tests
    "ANN401",              # Any is allowed for genuinely dynamic types
    "S311",                # pseudo-random generators are intentional in scientific code
    "S104",                # binding to 0.0.0.0 is intentional for a configurable server host
]
[tool.ruff.lint.per-file-ignores]
"**/__init__.py" = ["F401"]
"tests/**"       = ["D","PLR","ANN","S"]
"docs/conf.py"   = ["E402","UP031","ANN","S"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"          # enforces NumPy docstring style

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.isort]
known-first-party = ["precisionai"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true  # formats code blocks inside docstrings

[tool.pytest.ini_options]
addopts = "--cov=precisionai --cov-report=term-missing --cov-fail-under=90"

[tool.coverage.report]
fail_under = 90
exclude_lines = ["pragma: no cover","if __name__ == .__main__.:",
                 "raise ImportError","except ImportError"]

[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "standard"
reportMissingImports = false
reportMissingModuleSource = false
```

---

## pre-commit

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks: [check-added-large-files (--maxkb=1800), check-yaml, check-json,
            check-toml, end-of-file-fixer, trailing-whitespace,
            check-merge-conflict, detect-private-key, debug-statements,
            fix-byte-order-marker]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.17
    hooks: [ruff (--fix), ruff-format]

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: python -m pyright
        language: system
        types: [python]
        pass_filenames: false

  - repo: local
    hooks:
      - id: pytest
        entry: python -m pytest
        language: system
        args: [tests/, -q, --tb=short, --no-header]
        always_run: true
```

---

## Code style

### License header
- Every `.py` file starts with a two-line SPDX header, followed by a blank line before the module docstring (if any):
  ```python
  # Copyright 2026 Precision AI
  # SPDX-License-Identifier: Apache-2.0
  ```
- No "CONFIDENTIAL" or proprietary banners — this codebase is Apache 2.0 licensed; any such banner is a bug.

### Imports
- All imports at the **top of the file** — never inside functions
- Optional/unavailable deps: module-level `try/except ImportError` with a flag variable
- No silent fallbacks unless mathematically equivalent (document why)
- `__init__.py` re-exports only — no logic, no comments between import blocks
- `__all__` must be **alphabetically sorted**

```python
# optional dep pattern
try:
    import plotly.graph_objects as go  # type: ignore[import]
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# inside function that needs it
if not _PLOTLY_AVAILABLE:
    raise ImportError("plotly is required: pip install plotly") from None
```

### Docstrings
- **NumPy style** on all public functions, classes, and modules
- Module docstrings: plain English, no prefixes (no P0/P1/P2, no "original", no labels)
- One-line summary in **imperative mood** ("Compute …", "Return …", "Save …")
- Blank line between summary and extended description (D205)

### Type hints
- Required on **all** function signatures (public and private) — enforced by ruff (`ANN`) and pyright
- Use `X | Y` union syntax (Python 3.10+), not `Union[X, Y]` or `(X, Y)` in isinstance
- Use `Any` from `typing` for genuinely dynamic types — don't use `object` when methods will be called on it
- pyright config lives in `[tool.pyright]` in `pyproject.toml` — never pass type-checker flags inline

### Comments
- Only when the **why** is non-obvious
- No section-divider comments that describe what the code already says
- No cryptic labels, no TODO/FIXME without a ticket reference

### Exception handling
- Always `raise X from err` or `raise X from None` inside `except` blocks (B904)

### General
- `len()` returns `int` — never `int(len(...))`
- Loop variables not used in the body → rename to `_`
- `assert a and b` in tests → split into separate asserts
- `@pytest.fixture` not `@pytest.fixture()`
- `pytest.raises` always includes `match=` parameter
- No `# type: ignore[attr-defined]` when `Any` already covers the attribute

---

## Docs (Sphinx)

```
docs/
  conf.py          # version from git tag, logo, LaTeX/fancyhdr, enumitem fix
  index.rst        # toctree: readme, modules
  modules.rst      # autodoc for all public submodules
  requirements.txt # sphinx, sphinx-rtd-theme, myst-parser, sphinx-autodoc-typehints
  Makefile         # make html | make latexpdf | make clean
  assets/logo.png
```

- `conf.py` copies root `README.md` → `docs/readme.md` at build time
- Version read from `git describe --tags --exact-match`, falls back to `0.0.0`

---

## Testing

- Mirror package structure: `tests/test_<module>.py`
- Shared fixtures in `tests/conftest.py`
- 90% coverage hard minimum — enforced by pytest and pre-commit
- No mocks for database/filesystem unless truly unavoidable
- Integration tests marked `@pytest.mark.integration` and excluded from default runs

---

## Repository scaffolding

Every PAI Python project repository, in addition to the source layout above, ships with:

```
.gitattributes              # normalize line endings to LF; declare binary assets
CHANGELOG.md                # Keep a Changelog format + Semantic Versioning
CODE_OF_CONDUCT.md          # Contributor Covenant v2.1
MANIFEST.in                 # sdist packaging — include docs/license/changelog, exclude tests/examples
.github/
  dependabot.yml            # weekly pip + github-actions update checks
  ISSUE_TEMPLATE/
    bug_report.yml
    feature_request.yml
    config.yml               # blank_issues_enabled: false + link to SECURITY.md
  PULL_REQUEST_TEMPLATE.md
  workflows/
    ci.yml                   # dispatches to pre-commit.yml + unit-test.yml
    pre-commit.yml
    unit-test.yml
    release.yml               # tag push → build, PyPI publish, GitHub Release
```

These are not optional extras — treat them as part of the standard layout when auditing a project against this document.

## Releasing

- Versions are **never hardcoded** — `setuptools-scm` derives the package version from the current git tag (see the `pyproject.toml` block above). Tag with `vX.Y.Z`.
- Update `CHANGELOG.md` under `[Unreleased]` as you go; move those entries to a new `## [X.Y.Z] - YYYY-MM-DD` section when cutting a release.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`: runs the full test matrix, builds the sdist/wheel, verifies the built version matches the tag, publishes to PyPI via trusted publishing (`id-token: write`, no stored API tokens), and creates a GitHub Release with generated notes.
- CI checkout steps that build from git history (release builds, docs version detection) must use `fetch-depth: 0` — a shallow checkout has no tags for `setuptools-scm` or `git describe` to find.

---

## What to avoid

| Pattern | Instead |
|---|---|
| `int(len(x))` | `len(x)` |
| `isinstance(x, (A, B))` | `isinstance(x, A \| B)` |
| `assert a and b` (tests) | two separate asserts |
| `@pytest.fixture()` | `@pytest.fixture` |
| `pytest.raises(ValueError)` | `pytest.raises(ValueError, match="…")` |
| `raise X` inside except | `raise X from None` or `raise X from err` |
| Deferred imports inside functions | Module-level try/except with flag |
| Cryptic prefixes (P0–P4, "original") | Plain descriptive names |
| `# type: ignore[attr-defined]` on `Any` | Remove — redundant |
| `# noqa` suppression | Fix the underlying issue |
| Silent fallback for optional dep | Raise `ImportError` with install hint |
