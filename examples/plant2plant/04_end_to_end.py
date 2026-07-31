# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end demo: sample instance crops → embed → evaluate Plant→Plant.

Chains the same steps as ``01_sample_cluster_instances.py`` through
``03_run_eval.py`` in a single process, without intermediate JSON manifests —
a compact reference for using ``precisionai.agriemb`` to wire
``agristress-utils``, the PAI embedding API (https://embeddings.precision.ai/),
and ``agrieval`` together for the Plant→Plant retrieval scenario (how well
cropped plant instance embeddings cluster by crop/weed class label).

The dataset is verified before anything else and downloaded only if files are
missing, so re-running this on a complete dataset starts work immediately
instead of re-walking all 4950 files.

Class diversity depends on the chosen cluster — most L2 clusters contain a
single dominant plant label, some contain several. ``--cluster`` defaults to
``G2``, which carries three distinct labels; with a single-label cluster every
crop is a positive and the metrics saturate at 1.0000 without measuring
anything.

See also ``examples/image2image/04_end_to_end.py`` and
``examples/plant2image/04_end_to_end.py`` for the sibling scenarios.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/plant2plant/04_end_to_end.py
    python examples/plant2plant/04_end_to_end.py --cluster J1 --num-samples 6
    python examples/plant2plant/04_end_to_end.py --skip-download
"""

import argparse
import sys
from pathlib import Path

from openai import AuthenticationError

from precisionai.agriemb import (
    build_client,
    download,
    embed_images,
    print_evaluation,
    run_p2p_evaluation,
    sample_cluster_instances,
    verify_dataset,
)
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_MODEL, DEFAULT_REPO_ID
from precisionai.agristress.utils import AgriStressDataset, normalize_embeddings, validate_embeddings

DEFAULT_CLUSTER = "G2"


def main() -> None:
    """Run the full download-sample-embed-evaluate pipeline for Plant→Plant."""
    parser = argparse.ArgumentParser(
        description="End-to-end AgriStress-500 Plant2Plant demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    parser.add_argument("--skip-download", action="store_true", help="Never download; require a complete dataset")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help="L2 cluster to sample from")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of parent images to draw crops from")
    parser.add_argument(
        "--max-instances-per-image", type=int, default=3, help="Max instance crops to keep per parent image"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    args = parser.parse_args()

    # 1. Verify the dataset, downloading only when something is actually missing.
    print("[0/4] Verifying dataset...")
    missing = verify_dataset(args.dataset_dir)
    if missing and args.skip_download:
        raise RuntimeError(
            f"{len(missing)} dataset file(s) are missing from {args.dataset_dir} (e.g. {missing[:3]}), but "
            f"--skip-download was passed. Drop the flag, or run `python examples/01_download_dataset.py`."
        )
    if missing:
        print(f"[1/4] {len(missing)} file(s) missing; downloading (~12 GB in full, 20+ minutes)...")
        dataset_dir = download(args.dataset_dir, repo_id=args.repo_id)
    else:
        dataset_dir = Path(args.dataset_dir).resolve()
        print(f"[1/4] Dataset already complete at {dataset_dir}")
    dataset = AgriStressDataset(dataset_dir)

    # 2. Sample one cluster's instance crops.
    cluster, images, instances = sample_cluster_instances(
        dataset,
        cluster=args.cluster,
        num_samples=args.num_samples,
        max_instances_per_image=args.max_instances_per_image,
        seed=args.seed,
    )
    paths = dataset.resolve_paths(instances)
    print(f"[2/4] Sampled {len(instances)} instance crops from {len(images)} images in cluster {cluster}")

    # 3. Generate embeddings via the PAI embedding API.
    client = build_client()
    raw_embeddings = embed_images(client, paths, model=args.model)
    embeddings = normalize_embeddings(raw_embeddings)
    validate_embeddings(embeddings)
    print(f"[3/4] Generated {len(embeddings)} L2-normalized embeddings")

    # 4. Run Plant2Plant evaluation.
    result = run_p2p_evaluation(dataset, embeddings, k_values=args.k_values)
    print("[4/4] Plant2Plant evaluation:")
    print_evaluation(result)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/01_download_dataset.py` first.")
    except AuthenticationError:
        sys.exit("Error: the PAI embedding API rejected PAI_EMBEDDINGS_API_KEY. Check .env against .env.example.")
    except (ConnectionError, RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
