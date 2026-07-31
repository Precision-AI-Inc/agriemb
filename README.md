<p align="center">
  <img src="docs/assets/logo.png" alt="Precision AI Logo" width="120"/>
</p>

# Precision AI AgriEmb

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

---

Examples connecting parts of the Precision AI embedding public release: download the
AgriStress-500 dataset, sample a cluster, generate image embeddings through the hosted
PAI embedding API, and evaluate them with Image→Image, Plant→Image, and Plant→Plant
retrieval metrics — plus a Dense Patch Token (DPT) evaluation example built on real
tile-level embeddings from that same API.

This repo unifies three sibling projects into one pipeline:

| Project | Role | Package |
|---|---|---|
| [Precision AI Embeddings](https://embeddings.precision.ai/) | Hosted image embedding API (OpenAI SDK-compatible) | consumed via the `openai` SDK |
| [`agrieval`](https://github.com/Precision-AI-Inc/agrieval) | Embedding, Segmentation, and Dense Patch Token evaluation | `precisionai-agrieval` |
| [`agristress-utils`](https://github.com/Precision-AI-Inc/agristress-utils) | Dataset download, cluster enumeration, and payload assembly | `precisionai-agristress-utils` |

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
| `PAI_EMBEDDINGS_API_KEY` | An `sk-pai-` key minted at `<base-url>/keys` — for the hosted service, [embeddings.precision.ai/v1/keys](https://embeddings.precision.ai/v1/keys) (or the Portal's auto-provisioned Starter key) |

`.env` is gitignored — never commit real keys.

---

## The `precisionai.agriemb` package

The pipeline logic lives in `precisionai/agriemb/`, not just in the example scripts, so
it's reusable from your own code. Three retrieval scenarios are supported — Image→Image
(full field images against each other), Plant→Image (a field image against its own
cropped plant instances), and Plant→Plant (cropped plant instances against each other by
crop/weed class):

```python
from precisionai.agriemb import (
    build_client,
    download,
    embed_images,
    run_i2i_evaluation,
    sample_cluster_with_distractors,
)
from precisionai.agriemb.config import DEFAULT_DATASET_DIR
from precisionai.agristress.utils import AgriStressDataset, normalize_embeddings, validate_embeddings

# DEFAULT_DATASET_DIR is `examples/data/agristress/` inside this repo, so this
# reuses the copy `examples/01_download_dataset.py` already fetched instead of
# starting a second ~12 GB download somewhere else.
dataset_dir = download(DEFAULT_DATASET_DIR)
dataset = AgriStressDataset(dataset_dir)

# A query cluster plus distractors from other clusters, so the retrieval task
# has negatives it can actually rank wrongly (see Image→Image below).
cluster, image_ids, distractor_ids = sample_cluster_with_distractors(dataset, cluster="A1", num_samples=8)
paths = dataset.resolve_paths(image_ids + distractor_ids)

client = build_client()
embeddings = normalize_embeddings(embed_images(client, paths))
validate_embeddings(embeddings)

result = run_i2i_evaluation(dataset, embeddings, k_values=[5, 10])
```

| Module | Contents |
|---|---|
| `precisionai.agriemb.config` | Environment-driven defaults (`PAI_EMBEDDINGS_API_KEY`/`_BASE_URL`, dataset dir, model id, batch size) |
| `precisionai.agriemb.dataset` | `download()` (retrying, resumable), `verify_dataset()`, `sample_cluster()`, `sample_cluster_with_distractors()`, and `sample_cluster_instances()` built on `agristress-utils` |
| `precisionai.agriemb.embeddings` | `build_client()`, `image_to_data_uri()`, `embed_images()` for the PAI embedding API |
| `precisionai.agriemb.evaluate` | `run_i2i_evaluation()`, `run_p2i_evaluation()`, `run_p2p_evaluation()` — wire `AgriStressDataset`'s payload builders into `agrieval`'s `run_image2image_eval` / `run_plant2image_eval` / `run_plant2plant_eval`, plus `print_evaluation()` |
| `precisionai.agriemb.dpt` | `tile_image()`, `embed_patch_tokens()`, and `run_dpt_evaluation()` — see [Dense Patch Tokens](#dense-patch-token-dpt-evaluation) below |

---

## Examples

Download the dataset once — it's shared by every scenario below:

```bash
python examples/01_download_dataset.py         # -> examples/data/agristress/
```

**This is a ~12 GB download of 4950 files and takes 20+ minutes on a fast
connection.** HuggingFace rate-limits partway through often enough that the
script retries with backoff; if it still gives up it says so in one line, and
re-running resumes from whatever is already on disk. The scenario scripts below
verify the dataset before doing anything and skip the download when it is
already complete, so nothing re-walks all 4950 files on every run.

Each scenario then has its own directory under `examples/` with the same four-script
shape: a sampling step, a "generate the thing to evaluate" step, an evaluation step
(each writing a JSON artifact under `data/` that the next step reads), and a bundled
end-to-end script that chains all three in one process with no intermediate files —
plus a notebook version with inline image previews and (for the embedding scenarios)
interactive Plotly visualizations.

| Scenario | Directory | What it measures |
|---|---|---|
| Image→Image | [`examples/image2image/`](examples/image2image/) | How well one field image's embedding retrieves other images from the same L2 cluster, ranked against distractor images drawn from other clusters |
| Plant→Image | [`examples/plant2image/`](examples/plant2image/) | How well a field image's embedding retrieves its own cropped plant instances, and vice versa |
| Plant→Plant | [`examples/plant2plant/`](examples/plant2plant/) | How well cropped plant instance embeddings cluster by crop/weed class label |
| Dense Patch Tokens | [`examples/dpt/`](examples/dpt/) | Per-patch feature-map geometry of a dense vision backbone (built from real tile-level embeddings — see below) |

```bash
# Stepped, e.g. Image2Image (same shape for plant2image/ and plant2plant/):
python examples/image2image/01_sample_cluster.py           # -> data/image2image_sample_manifest.json
python examples/image2image/02_generate_embeddings.py      # -> data/image2image_embeddings.json
python examples/image2image/03_run_eval.py                 # -> data/image2image_result.json

# Or bundled end-to-end, no intermediate files:
python examples/image2image/04_end_to_end.py
python examples/image2image/04_end_to_end.py --cluster C1 --num-samples 12
python examples/image2image/04_end_to_end.py --skip-download   # fail instead of downloading
```

Every scenario defaults to a cluster that is known to produce meaningful
numbers rather than to a random one. AgriStress-500's clusters are mostly tiny
(104 of 121 hold fewer than 8 images) and a third own no instance crops or
cannot be parsed at all, so a random cluster usually yields a wall of perfect
`1.0000` scores that measure nothing — or no result at all. The library warns
when a sample is capped, when requested k values get clamped, and when every
evaluated item shares one class.

### What Image→Image measures

Image→Image counts same-cluster images as relevant. Sampling from one cluster
alone therefore leaves *no* negatives: every candidate is a positive, so
precision and nDCG come out at 1.0000 by construction whatever the embeddings
do. `01_sample_cluster.py` samples the query cluster **plus distractors from
other clusters** (6 clusters x 2 images by default), so the ranking has
something it can get wrong and the reported numbers reflect embedding quality.
Pass `--num-distractor-clusters 0` to see the degenerate single-cluster
behaviour for yourself.

Each script accepts `--help` for its full option list (cluster choice, sample size,
k-values, output paths, etc.). Each scenario directory also has a notebook at
`04_end_to_end.ipynb` (requires `pip install -e ".[notebook]"`):

```bash
jupyter notebook examples/image2image/04_end_to_end.ipynb
```

### Dense Patch Token (DPT) evaluation

DPT scores the per-patch feature-map output of a dense vision backbone — the spatial
grid of patch embeddings a model produces before pooling into a single image
embedding. That's a different kind of artifact from the pooled, whole-image vectors
the hosted PAI embedding API returns directly, so **no dense-feature backbone is wired
up to that API**: instead, `examples/dpt/` slices each sampled image into a grid of
tiles and embeds *every tile* individually through the real API, then stacks the
resulting real per-tile vectors into a `[P, H, W]` feature map. Every value in that
feature map is a genuine embedding from the hosted API — the tiling scheme is the only
fabricated part, not the features. It requires `.env` / PAI API credentials like the
other scenarios, and since each tile is a separate `embeddings.create` call, total API
calls scale as `num_samples * grid_height * grid_width`, which is an 8x8 grid over
2 images by default: **128 calls per run**.

The 8x8 grid is the smallest that keeps the label-aware metrics working. Coarser
grids are cheaper, but majority-vote downsampling of the ground-truth masks then
resolves every patch to `background` on most clusters, leaving silhouette,
Calinski-Harabasz, ARI, NMI and PC1-AUROC all reporting `N/A` and k-NN purity
trivially at 1.0000. `run_dpt_evaluation` warns loudly if that happens anyway.

Label-aware metrics (on by default) additionally validate each image's
full-resolution ground-truth mask, costing roughly 15 s per image. Measured on the
two-image default: **34 s** label-aware versus **4 s** with `--no-labels` (or
`WITH_LABELS = False` in the notebook).

---

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (> 1800 KB), trailing whitespace, merge conflicts, private keys, debug statements, BOM removal |
| `ruff-check` | Linting and import sorting (auto-fix) |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

Tests live in `tests/` and are pure and offline (no dataset, no API):

```bash
pytest
```

---

## Project layout

```
precisionai/agriemb/
  config.py               # Environment-driven configuration (dataset dir anchored to examples/, not cwd)
  dataset.py              # Dataset download/verify + cluster/instance/distractor sampling (agristress-utils)
  embeddings.py           # PAI embedding API client + batch embedding (file paths or in-memory tiles)
  evaluate.py             # Image2Image, Plant2Image, Plant2Plant evaluation (agrieval)
  dpt.py                  # Tile + embed images into patch-token feature maps; Dense Patch Token evaluation (agrieval)
docs/                     # Sphinx (HTML + LaTeX/PDF)
tests/                    # Unit tests for sampling, verification, encoding and tiling (no network)
examples/                 # Standalone runnable scripts + notebooks
  01_download_dataset.py  # Shared prerequisite for every scenario below
  image2image/            # Sample -> embed -> evaluate -> end-to-end, plus a notebook
  plant2image/            # Sample -> embed -> evaluate -> end-to-end, plus a notebook
  plant2plant/            # Sample -> embed -> evaluate -> end-to-end, plus a notebook
  dpt/                    # Sample -> tile+embed -> evaluate -> end-to-end, plus a notebook
```

`precisionai` itself is a PEP 420 namespace package shared with the sibling
`precisionai-agristress-utils` and `precisionai-agrieval` distributions, so it
deliberately has **no** `__init__.py`. Adding one turns it into a regular
package, which hides whichever sibling is not found first on `sys.path`.

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

Push a tag matching `v*.*.*` (e.g. `v0.1.0`) to trigger `.github/workflows/release.yml`, which builds the sdist/wheel and creates a GitHub Release. The package version is derived from the git tag via `setuptools-scm` — update `CHANGELOG.md` before tagging.

---

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching, and PR guidelines, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. [CLAUDE.md](CLAUDE.md) documents the code style and conventions enforced in this repo.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md).

See [CHANGELOG.md](CHANGELOG.md) for release history.
