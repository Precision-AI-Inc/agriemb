# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end demo: sample a cluster → embed → evaluate Image→Image.

Chains the same steps as ``01_sample_cluster.py`` through ``03_run_eval.py``
in a single process, without intermediate JSON manifests — a compact
reference for using ``precisionai.agriemb`` to wire ``agristress-utils``, the
PAI embedding API (https://embeddings.precision.ai/), and ``agrieval``
together for the Image→Image retrieval scenario (how well one field image's
embedding retrieves other images from the same L2 cluster).

See also ``examples/plant2image/04_end_to_end.py`` and
``examples/plant2plant/04_end_to_end.py`` for the sibling scenarios.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/image2image/04_end_to_end.py
    python examples/image2image/04_end_to_end.py --cluster A1 --num-samples 12
"""

import argparse

from precisionai.agriemb import build_client, download, embed_images, run_i2i_evaluation, sample_cluster
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_MODEL, DEFAULT_REPO_ID
from precisionai.agrieval.emb import print_result
from precisionai.agristress.utils import AgriStressDataset, normalize_embeddings, validate_embeddings


def main() -> None:
    """Run the full download-sample-embed-evaluate pipeline for Image→Image."""
    parser = argparse.ArgumentParser(description="End-to-end AgriStress-500 Image2Image demo")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    parser.add_argument("--cluster", default=None, help="L2 cluster to sample from (default: random)")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    args = parser.parse_args()

    # 1. Download the dataset.
    dataset_dir = download(args.dataset_dir, repo_id=args.repo_id)
    dataset = AgriStressDataset(dataset_dir)
    print(f"[1/4] Downloaded dataset to {dataset_dir}")

    # 2. Sample one cluster.
    cluster, sample = sample_cluster(dataset, cluster=args.cluster, num_samples=args.num_samples, seed=args.seed)
    paths = dataset.resolve_paths(sample)
    print(f"[2/4] Sampled {len(sample)} images from cluster {cluster}")

    # 3. Generate embeddings via the PAI embedding API.
    client = build_client()
    raw_embeddings = embed_images(client, paths, model=args.model)
    embeddings = normalize_embeddings(raw_embeddings)
    validate_embeddings(embeddings)
    print(f"[3/4] Generated {len(embeddings)} L2-normalized embeddings")

    # 4. Run Image2Image evaluation.
    result = run_i2i_evaluation(dataset, embeddings, k_values=args.k_values)
    print("[4/4] Image2Image evaluation:\n")
    print_result(result)


if __name__ == "__main__":
    main()
