# Precision AI — Python Project Standards

Apply these standards when writing, reviewing, or refactoring code in any PAI Python project.

This repo (`agriemb`) unifies three sibling projects — `agristress-utils`, the PAI
embedding API (https://embeddings.precision.ai/), and `agrieval` — into one
download-sample-embed-evaluate pipeline. It is a library + examples repo, not an API
service.

---

## AgriStress-500: what the data actually looks like

Measured against the full 4950-file dataset. These numbers drive the example
defaults; a random cluster is almost never a usable one.

- **121 L2 clusters, 451 images, 4041 instance crops, 451 masks.** Cluster sizes are
  heavily skewed: `{1: 38, 2: 26, 3: 19, 4: 8, 5: 7, 6: 3, 7: 3, 8: 4, 9: 4, 11: 1,
  12: 1, 13: 2, 14: 2, 16: 2, 21: 1}`. Only 17 of 121 clusters hold 8 or more images.
  Largest: `C1` (21), `A1` (16), `B1` (16), `A2` (14), `I1` (14).
- **27 clusters own no instance crops at all**, so Plant→Image and Plant→Plant have
  nothing to retrieve there.
- **14 clusters have a null `camera_source`**, which `agristress-utils` cannot parse
  (`ImageAttributes.camera_source` is typed as a required `str`): `B1 B2 B3 B4 B5 I1
  S1 S2 S3 S4 T1 T2 T3 T4`. **This needs fixing upstream as `str | None`.** Until then
  only 80 of 121 clusters are usable for the plant scenarios.
- **`G2` is the only cluster with three distinct plant labels** (`Crop | Barley`,
  `Weed | Broadleaf`, `Weed | Morning Glory`); almost every other cluster is
  single-label, which makes Plant→Plant metrics saturate.
- **Masks are mostly background at coarse patch grids.** With 2 images at seed 0, only
  26 of 121 clusters yield more than one patch class at 8x8, and far fewer at 4x4.
  Best-balanced at 8x8: `L1` (86 crop / 42 background), `A4` (96/32), `G1` (32/96),
  `A1` (26/102); `E6` is the only one with three classes. `B1`, `C1` and `Q3` are
  all-background even at 8x8, so they are useless for DPT label-aware metrics.
- **Image→Image positives are same-cluster images**, so a single-cluster sample has no
  negatives and every precision/nDCG figure is 1.0 by construction. Always sample
  distractors. Plant→Image relevance is parent↔instance instead, so a single path
  class there is normal and *not* a degeneracy.

## Verifying a change to the examples

Lint passing is not evidence: this repo once passed 13/13 pre-commit hooks while three
documented commands crashed and a fourth printed a wall of meaningless `1.0000`s. Run
each scenario's `01 → 02 → 03` chain and its `04_end_to_end.py` with no arguments from a
clean `data/`, and read the numbers. Default run cost: 10 + 8 + 9 + 128 = 155
`embeddings.create` calls, most of them DPT tiles.

---

## Environment

- Python 3.10+, managed with `python -m venv .venv` (never conda)
- Install: `pip install -e ".[dev]"` → installs all deps including pre-commit and pytest
- Register hooks once: `pre-commit install`

---

## Project layout

```
precisionai/agriemb/
  config.py      # env-var configuration (API key/base URL, dataset dir, model, batch size)
  dataset.py     # download(), verify_dataset(), sample_cluster*() — built on agristress-utils
  embeddings.py  # build_client() + embed_images() — the PAI embedding API (openai SDK)
  evaluate.py    # run_i2i_evaluation(), run_p2i_evaluation(), run_p2p_evaluation(), print_evaluation() — wire agristress-utils payloads into agrieval
  dpt.py         # tile_image(), embed_patch_tokens(), run_dpt_evaluation() — Dense Patch Token feature maps + evaluation (agrieval)
docs/            # Sphinx (HTML + LaTeX/PDF)
tests/           # offline unit tests (no dataset, no network); run with `pytest`
examples/        # standalone runnable scripts + notebooks, one scenario dir per retrieval task, built on precisionai.agriemb
```

`precisionai` is a fixed, namespace-only top-level package shared by every PAI Python
project — it is what makes `precisionai.<namespace>.*` imports consistent across the org.
It is a **PEP 420 namespace package and must contain no `precisionai/__init__.py`**, not
even one holding only an SPDX header. An `__init__.py` turns it into a regular package,
so whichever distribution is found first on `sys.path` hides every sibling's subpackages;
it also makes several distributions co-own one file, so uninstalling any one of them
breaks the others.

---

## pyproject.toml — canonical config

```toml
[project]
dynamic = ["version"]   # version comes from the git tag via setuptools-scm, never hardcoded

[tool.setuptools.packages.find]
namespaces = true             # `precisionai` is a PEP 420 namespace package

[tool.pytest.ini_options]
testpaths = ["tests"]         # so a bare `pytest` collects the offline suite only

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
    "SIM108",  # ternary readability
    "D100","D104",          # module/package docstrings optional
    "D203","D213",          # pydocstyle conflicts — always ignore these two
    "PLR0913","PLR2004",    # arg count + magic values common in metrics
    "ANN401",              # Any is allowed for genuinely dynamic types
    "S311",                # pseudo-random generators are intentional in scientific code
]
[tool.ruff.lint.per-file-ignores]
"**/__init__.py" = ["F401"]
"docs/conf.py"   = ["E402","UP031","ANN","S"]
"tests/**"       = ["S101","D102"]        # `assert` is the point; test names document themselves

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
    hooks: [ruff-check (--fix), ruff-format]   # `ruff` is a legacy alias for `ruff-check`

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: python -m pyright
        language: system
        types: [python]
        pass_filenames: false
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

- The `html` target passes `-W --keep-going`, so an autodoc import failure fails the build. Without it a broken build still exits 0 and ships an API chapter with no signatures in it.
- `conf.py` copies root `README.md` → `docs/readme.md` at build time
- Version read from `git describe --tags --exact-match`, falls back to `0.0.0`

---

## Repository scaffolding

Every PAI Python project repository, in addition to the source layout above, ships with:

```
.gitattributes              # normalize line endings to LF; declare binary assets
AGENTS.md                   # short pointer to this file; a regular file, never a symlink
CHANGELOG.md                # Keep a Changelog format + Semantic Versioning
CODE_OF_CONDUCT.md          # Contributor Covenant v2.1
MANIFEST.in                 # sdist packaging — include license/readme/changelog, prune examples, tests, docs and .github
.github/
  dependabot.yml            # weekly pip + github-actions update checks
  ISSUE_TEMPLATE/
    bug_report.yml
    feature_request.yml
    config.yml               # blank_issues_enabled: false + link to SECURITY.md
  PULL_REQUEST_TEMPLATE.md
  workflows/
    ci.yml                    # dispatches to pre-commit.yml, test.yml and docs.yml
    pre-commit.yml
    test.yml                  # import smoke-test from the repo root + pytest
    docs.yml                  # strict Sphinx build; fails on an empty API reference
    release.yml               # tag push → build, GitHub Release
```

These are not optional extras — treat them as part of the standard layout when auditing a project against this document.

## Releasing

- Versions are **never hardcoded** — `setuptools-scm` derives the package version from the current git tag (see the `pyproject.toml` block above). Tag with `vX.Y.Z`.
- Update `CHANGELOG.md` under `[Unreleased]` as you go; move those entries to a new `## [X.Y.Z] - YYYY-MM-DD` section when cutting a release.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`: builds the sdist/wheel, verifies the built version matches the tag, and creates a GitHub Release (named after the tag, e.g. `v1.0.0`) with the built artifacts attached and generated release notes. This repo is not published to PyPI.
- The release workflow must verify the artifacts in `dist/` and install the built wheel. Reading the version from `importlib.metadata` in the build checkout resolves through the stray `*.egg-info/` that `python -m build` leaves behind, so it passes without checking anything.
- The release workflow must also run the same checks `ci.yml` does (pre-commit, tests, docs) before it builds anything. A tag can be pushed from any commit, and `ci.yml` only ever sees pushes to the default branch and pull requests, so nothing else stops a red commit from being published.
- CI checkout steps that build from git history (release builds, docs version detection) must use `fetch-depth: 0` — a shallow checkout has no tags for `setuptools-scm` or `git describe` to find.

---

## What to avoid

| Pattern | Instead |
|---|---|
| `int(len(x))` | `len(x)` |
| `isinstance(x, (A, B))` | `isinstance(x, A \| B)` |
| `raise X` inside except | `raise X from None` or `raise X from err` |
| Deferred imports inside functions | Module-level try/except with flag |
| Cryptic prefixes (P0–P4, "original") | Plain descriptive names |
| `# type: ignore[attr-defined]` on `Any` | Remove — redundant |
| `# noqa` suppression | Fix the underlying issue |
| Silent fallback for optional dep | Raise `ImportError` with install hint |
