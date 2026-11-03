# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `precisionai.agriemb` package: `dataset` (download + cluster/instance sampling),
  `embeddings` (PAI embedding API client, accepting file paths or in-memory tile crops),
  `evaluate` (Image→Image, Plant→Image, and Plant→Plant evaluation), and `dpt` (tile an
  image and embed every tile through the real PAI embedding API to build a patch-token
  feature map, paired with the dataset's real ground-truth masks for Dense Patch Token
  evaluation), unifying `agristress-utils`, the PAI embedding API
  (https://embeddings.precision.ai/), and `agrieval`.
- Standalone example scripts under `examples/`: a shared `01_download_dataset.py`, and a
  `sample` → `generate` → `run_eval` → bundled `end_to_end` script set (plus a Jupyter
  notebook) per scenario in `examples/image2image/`, `examples/plant2image/`,
  `examples/plant2plant/`, and `examples/dpt/`.
- `sample_cluster_with_distractors()`: samples a query cluster plus images from other
  clusters, so Image→Image retrieval has negatives to rank below the positives.
- `verify_dataset()`: reports which manifest-referenced files are missing on disk, so the
  example scripts can skip a completed download instead of re-walking all 4950 files, and
  can detect the half-downloaded state an interrupted run leaves behind.
- `print_evaluation()`: prints an evaluation result with undefined metrics reported as
  `n/a` instead of raising `TypeError` out of `agrieval`'s `:.4f` formatting.
- `DEFAULT_GRID_SIZE`: the default Dense Patch Token grid, re-exported from
  `precisionai.agriemb` so the DPT scripts and notebook take it from the package root
  alongside the rest of the public API instead of reaching into `precisionai.agriemb.dpt`.
- `--skip-download` on all four `04_end_to_end.py` scripts, and a `[0/4] Verifying
  dataset...` step so the wait is never silent.
- `tests/`: offline unit tests for cluster sampling, dataset verification, data-URI
  encoding and DPT tiling, plus CI jobs that run them, import the package from the repo
  root, and build the docs.

### Fixed

- `LICENSE.md` now carries the exact Apache License 2.0 text. The copy inherited from
  the project template had several clauses reworded or cut short, so GitHub could not
  identify the license, and the package shipped a license that was not quite Apache 2.0.
- `DEFAULT_DATASET_DIR` is now anchored to this repo's `examples/` directory instead of
  the caller's current working directory. Previously, running a script or notebook from
  within its own scenario subdirectory (the default for Jupyter/VS Code notebooks) could
  trigger a fresh ~12 GB re-download of AgriStress-500 nested under that subdirectory
  instead of reusing the one already-downloaded copy.
- `examples/dpt/` defaults raised from a 4x4 tile grid to 8x8, taking default
  `embeddings.create` calls from 32 to 128 per run (2 images x 8 x 8). The 4x4 grid was
  cheaper but broken: majority-vote downsampling of the ground-truth masks resolved every
  patch to `background` on most clusters, so silhouette, Calinski-Harabasz, ARI, NMI and
  PC1-AUROC all reported `N/A` and k-NN purity was trivially 1.0000. An earlier entry
  described that cut as an improvement and miscounted it as 144 → 16 calls; the real 4x4
  default was 32 calls, and the feature it disabled is the headline of the DPT scenario.
  `run_dpt_evaluation()` now also warns when patch labels collapse to a single class.
- `precisionai/__init__.py` deleted and `[tool.setuptools.packages.find] namespaces` set
  to `true`. That file made `precisionai` a regular package rather than the PEP 420
  namespace package it is documented to be, so with the repo root on `sys.path` the
  sibling `precisionai.agristress` and `precisionai.agrieval` packages became invisible:
  the README's own library snippet failed from the repo root, and every autodoc directive
  failed while the Sphinx build still exited 0 with an empty API chapter. **The same file
  ships in `precisionai-agrieval` and `precisionai-agristress-utils` and needs the same
  removal there.**
- `docs/Makefile`'s `html` target passes `-W --keep-going`, so an autodoc import failure
  is now fatal rather than a warning under a green build.
- `download()` retries transient network failures with exponential backoff and, when it
  gives up, raises a one-line "Download interrupted after N attempts. It is resumable -
  re-run this script." naming the real underlying error, instead of a 40-line traceback.
  The download is resumable, which nothing previously said.
- `sample_cluster_instances()` rejects clusters with no instance crops (27 of 121) and
  re-raises the pydantic `ValidationError` from clusters with a null `camera_source`
  (14 of 121) as a message naming usable clusters. Previously the first produced a
  `TypeError` deep inside result formatting and the second a raw pydantic traceback.
  **The underlying fix belongs upstream in `agristress-utils`: `ImageAttributes.camera_source`
  should be typed `str | None`.**
- Every `03_run_eval.py` writes its result JSON *before* printing the summary, so a
  formatting failure can no longer discard a completed evaluation.
- Example scripts default to clusters that produce meaningful metrics instead of a random
  cluster, matching what the notebooks already hardcoded. 104 of 121 clusters hold fewer
  than 8 images, so the previous defaults deterministically produced a wall of `1.0000`
  scores. `sample_cluster()` now warns when the sample is capped, and the evaluation
  functions warn when requested k values are clamped or every item shares one class.
- `embed_images()` shows a progress bar; a dozen full-resolution images previously took
  over a minute with no output at all.
- Example scripts turn missing credentials, rejected API keys and missing input artifacts
  into a one-line actionable message and exit 1, instead of a raw traceback.
- CI runs on pushes to `main` as well as pull requests (its only job was gated to pull
  requests, so pushes reported green having run nothing) and asks for `contents: read`
  rather than `contents: write`.
- The release workflow verifies the artifacts in `dist/` and installs the built wheel,
  instead of reading a version that only resolved through a stray `*.egg-info/` left in
  the working directory.
- The release workflow runs pre-commit, the tests and the docs build before it builds
  anything, so a tag pushed from a commit whose checks are red can no longer publish a
  GitHub Release. A tag can be pushed from any commit, and `ci.yml` only ever saw pushes
  to `main` and pull requests.
- Documented DPT runtimes replaced with measured ones: ~15 s per image label-aware
  (34 s for the two-image default), 4 s with `--no-labels`. The previous "on the order of
  a minute per image" and "near-instant" were both wrong.
- README corrections: the dataset lands in `examples/data/agristress/` (not
  `data/agristress/`), the library snippet uses `DEFAULT_DATASET_DIR` so it cannot start a
  second ~12 GB download, the download size and duration are stated, and the leftover
  "now a 4x4" phrasing, ragged layout comments and two spellings of the key-minting URL
  are fixed.
- `.pre-commit-config.yaml` uses the `ruff-check` hook id; `ruff` is a legacy alias in
  current `ruff-pre-commit`.
- `argparse.ArgumentDefaultsHelpFormatter` on every example script, so `--help` shows the
  defaults; the DPT tile-count summary is no longer split across two `print` calls; and
  the k-NN progress bar is terminated before the evaluation summary.

[Unreleased]: https://github.com/Precision-AI-Inc/agriemb/commits/main
