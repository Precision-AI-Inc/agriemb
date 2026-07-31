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

### Fixed

- `DEFAULT_DATASET_DIR` is now anchored to this repo's `examples/` directory instead of
  the caller's current working directory. Previously, running a script or notebook from
  within its own scenario subdirectory (the default for Jupyter/VS Code notebooks) could
  trigger a fresh ~13 GB re-download of AgriStress-500 nested under that subdirectory
  instead of reusing the one already-downloaded copy.
- `examples/dpt/` defaults cut from a 12x12 tile grid to 4x4 (and the notebook's
  `NUM_SAMPLES` from 4 to 2, matching its own doc comment), reducing default
  `embeddings.create` calls from 144 to 16 per run. The larger defaults could exhaust API
  quota on a single run.

[Unreleased]: https://github.com/Precision-AI-Inc/agriemb/compare/v0.1.0...HEAD
