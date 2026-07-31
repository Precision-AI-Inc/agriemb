# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end demo: sample images → tile + embed patch tokens → evaluate Dense Patch Tokens.

Chains the same steps as ``01_sample_cluster.py`` through ``03_run_eval.py``
in a single process, without intermediate JSON manifests — a compact
reference for using ``precisionai.agriemb`` to wire ``agristress-utils``,
the PAI embedding API (https://embeddings.precision.ai/), and ``agrieval``
together for Dense Patch Token (DPT) evaluation, which scores the per-patch
feature-map output of a dense vision backbone (the spatial grid of patch
embeddings before pooling into a single image embedding).

No dense-feature backbone is wired up to the hosted PAI embedding API (it
returns one pooled vector per image, not a spatial patch grid), so each
sampled image is sliced into a grid of tiles and every tile is embedded
individually through the real API — the resulting ``[P, H, W]`` feature map
is built entirely from genuine embeddings; only the tiling scheme is
fabricated, not the feature values. Each tile is a separate
``embeddings.create`` call, so total API calls scale as
``num_samples * grid_height * grid_width`` — keep the grid modest.

By default this also validates each sampled image's full-resolution
ground-truth mask for label-aware metrics, which costs on the order of a
minute per image — pass ``--no-labels`` to skip masks entirely for a
near-instant, unsupervised-metrics-only result.

See also ``examples/image2image/04_end_to_end.py``,
``examples/plant2image/04_end_to_end.py``, and
``examples/plant2plant/04_end_to_end.py`` for the sibling scenarios.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/dpt/04_end_to_end.py
    python examples/dpt/04_end_to_end.py --cluster B1 --num-samples 2
    python examples/dpt/04_end_to_end.py --no-labels
"""

import argparse
from pathlib import Path

from precisionai.agriemb import build_client, download, embed_patch_tokens, run_dpt_evaluation, sample_cluster
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_MODEL, DEFAULT_REPO_ID
from precisionai.agrieval.dpt import print_result
from precisionai.agristress.utils import AgriStressDataset


def main() -> None:
    """Run the full download-sample-tile-embed-evaluate pipeline for Dense Patch Tokens."""
    parser = argparse.ArgumentParser(description="End-to-end AgriStress-500 DPT demo")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    parser.add_argument("--cluster", default=None, help="L2 cluster to sample from (default: random)")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of images to sample")
    parser.add_argument("--grid-height", type=int, default=4, help="Patch grid height (H), in tiles")
    parser.add_argument("--grid-width", type=int, default=4, help="Patch grid width (W), in tiles")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip ground-truth masks entirely for a near-instant, unsupervised-metrics-only result",
    )
    args = parser.parse_args()

    # 1. Download the dataset.
    dataset_dir = download(args.dataset_dir, repo_id=args.repo_id)
    dataset = AgriStressDataset(dataset_dir)
    print(f"[1/4] Downloaded dataset to {dataset_dir}")

    # 2. Sample one cluster's images.
    cluster, sample = sample_cluster(dataset, cluster=args.cluster, num_samples=args.num_samples, seed=args.seed)
    paths = dataset.resolve_paths(sample)
    stems = {Path(image_id).stem: path for image_id, path in paths.items()}
    print(f"[2/4] Sampled {len(sample)} images from cluster {cluster}")

    # 3. Tile each image and embed every tile via the PAI embedding API.
    client = build_client()
    images = embed_patch_tokens(client, stems, grid_size=(args.grid_height, args.grid_width), model=args.model)
    n_tiles = len(stems) * args.grid_height * args.grid_width
    print(f"[3/4] Embedded {n_tiles} tiles into [H={args.grid_height}, W={args.grid_width}] feature maps")

    # 4. Run Dense Patch Token evaluation.
    result = run_dpt_evaluation(dataset_dir, images, k_values=args.k_values, with_labels=not args.no_labels)
    print("[4/4] Dense Patch Token evaluation:\n")
    print_result(result)


if __name__ == "__main__":
    main()
