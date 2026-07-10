# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `precisionai.agriemb` package: `dataset` (download + cluster sampling), `embeddings`
  (PAI embedding API client), and `evaluate` (Image→Image evaluation), unifying
  `agristress-utils`, the PAI embedding API (https://embeddings.precision.ai/), and
  `agrieval`.
- Standalone example scripts (`examples/01_download_dataset.py` through
  `examples/05_end_to_end.py`) and a Jupyter notebook (`examples/end_to_end.ipynb`).

[Unreleased]: https://github.com/Precision-AI-Inc/agriemb/compare/v0.1.0...HEAD
