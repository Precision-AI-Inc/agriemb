<p align="center">
  <img src="https://raw.githubusercontent.com/Precision-AI-Inc/agriemb/main/docs/assets/logo.png" alt="Precision AI Logo" width="120"/>
</p>

# PAI AgriEmb

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/precisionai-agriemb.svg?include_prereleases)](https://pypi.org/project/precisionai-agriemb/)
[![Python](https://img.shields.io/pypi/pyversions/precisionai-agriemb.svg?include_prereleases)](https://pypi.org/project/precisionai-agriemb/)

---

Examples connecting parts of the Precision AI embedding public release: download the
AgriStress-500 dataset, sample a cluster, generate image embeddings through the hosted
PAI embedding API, and evaluate them with Image→Image retrieval metrics.

This repo unifies three sibling projects into one pipeline:

| Project | Role | Package |
|---|---|---|
| [`agristress-utils`](https://github.com/Precision-AI-Inc/agristress-utils) | Dataset download, cluster enumeration, and payload assembly | `precisionai-agristress-utils` |
| [Precision AI Embeddings](https://embeddings.precision.ai/) | Hosted image embedding API (OpenAI SDK-compatible) | consumed via the `openai` SDK |
| [`agrieval`](https://github.com/Precision-AI-Inc/agrieval) | Embedding evaluation (retrieval metrics) | `precisionai-agrieval` |

Coding standards, naming conventions, and tooling configuration are governed by
[CLAUDE.md](CLAUDE.md).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Runtime only
pip install -r requirements.txt

# Full development (pre-commit, type checking)
pip install -e ".[dev]"
pre-commit install

# To also run the Jupyter notebook and visualizations
pip install -e ".[notebook]"
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your PAI embedding API credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `PAI_EMBEDDINGS_BASE_URL` | Base URL of the PAI embedding API, ending in `/v1` — defaults to the hosted service at [embeddings.precision.ai](https://embeddings.precision.ai/) |
| `PAI_EMBEDDINGS_API_KEY` | An `sk-pai-` key minted at `<base-url>/keys` (or the Portal's Starter key) |

`.env` is gitignored — never commit real keys.

---

## The `precisionai.agriemb` package

The pipeline logic lives in `precisionai/agriemb/`, not just in the example scripts, so
it's reusable from your own code:

```python
from precisionai.agriemb import build_client, download, embed_images, run_i2i_evaluation, sample_cluster
from precisionai.agristress.utils import AgriStressDataset, normalize_embeddings, validate_embeddings

dataset_dir = download("data/agristress")
dataset = AgriStressDataset(dataset_dir)

cluster, image_ids = sample_cluster(dataset, num_samples=8)
paths = dataset.resolve_paths(image_ids)

client = build_client()
embeddings = normalize_embeddings(embed_images(client, paths))
validate_embeddings(embeddings)

result = run_i2i_evaluation(dataset, embeddings, k_values=[5, 10])
```

| Module | Contents |
|---|---|
| `precisionai.agriemb.config` | Environment-driven defaults (`PAI_EMBEDDINGS_API_KEY`/`_BASE_URL`, dataset dir, model id, batch size) |
| `precisionai.agriemb.dataset` | `download()` and `sample_cluster()` built on `agristress-utils` |
| `precisionai.agriemb.embeddings` | `build_client()`, `image_to_data_uri()`, `embed_images()` for the PAI embedding API |
| `precisionai.agriemb.evaluate` | `run_i2i_evaluation()` — wires `AgriStressDataset.build_i2i_payload` into `agrieval`'s `run_image2image_eval` |

---

## Examples

Run in order from the repo root — each stage writes a JSON artifact under `data/` that
the next stage reads:

```bash
python examples/01_download_dataset.py        # -> data/agristress/
python examples/02_sample_cluster.py           # -> data/sample_manifest.json
python examples/03_generate_embeddings.py      # -> data/embeddings.json
python examples/04_run_image2image_eval.py     # -> data/image2image_result.json
```

Or run the whole pipeline in one process (no intermediate files):

```bash
python examples/05_end_to_end.py
python examples/05_end_to_end.py --cluster A1 --num-samples 12
```

Each script accepts `--help` for its full option list (cluster choice, sample size,
k-values, output paths, etc.).

A notebook version with inline image previews and interactive Plotly visualizations is
at [`examples/end_to_end.ipynb`](examples/end_to_end.ipynb) (requires
`pip install -e ".[notebook]"`):

```bash
jupyter notebook examples/end_to_end.ipynb
```

---

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (> 1800 KB), trailing whitespace, merge conflicts, private keys, debug statements, BOM removal |
| `ruff` | Linting and import sorting (auto-fix) |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

---

## Project layout

```
precisionai/agriemb/
  config.py         # Environment-driven configuration
  dataset.py         # Dataset download + cluster sampling (agristress-utils)
  embeddings.py       # PAI embedding API client + batch embedding
  evaluate.py         # Image2Image evaluation (agrieval)
docs/                # Sphinx (HTML + LaTeX/PDF)
examples/            # Standalone runnable scripts + notebook
```

See [CLAUDE.md](CLAUDE.md) for the full coding standard covering imports, docstrings,
type hints, and what to avoid.

---

## Documentation

```bash
cd docs
make html       # HTML docs → docs/_build/html/index.html
make latexpdf   # PDF      → docs/_build/latex/documentation.pdf
make clean      # Remove build artefacts
```

Dependencies: `pip install -r docs/requirements.txt`

---

## Releasing

Push a tag matching `v*.*.*` (e.g. `v0.1.0`) to trigger `.github/workflows/release.yml`, which builds the sdist/wheel, publishes to PyPI via trusted publishing, and creates a GitHub Release. The package version is derived from the git tag via `setuptools-scm` — update `CHANGELOG.md` before tagging.

---

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching, and PR guidelines, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. [CLAUDE.md](CLAUDE.md) documents the code style and conventions enforced in this repo.

---

## License

[Apache 2.0](LICENSE.md) © Precision AI
