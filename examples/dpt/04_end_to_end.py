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
``num_samples * grid_height * grid_width``: 2 x 8 x 8 = 128 by default.

The dataset is verified before anything else and downloaded only if files are
missing, so re-running this on a complete dataset starts work immediately
instead of re-walking all 4950 files.

By default this also validates each sampled image's full-resolution
ground-truth mask for label-aware metrics, costing roughly 15 s per image —
pass ``--no-labels`` to skip masks entirely for an unsupervised-metrics-only
result. Measured on the two-image default: 34 s label-aware, 4 s without.

See also ``examples/image2image/04_end_to_end.py``,
``examples/plant2image/04_end_to_end.py``, and
``examples/plant2plant/04_end_to_end.py`` for the sibling scenarios.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/dpt/04_end_to_end.py
    python examples/dpt/04_end_to_end.py --cluster A4 --num-samples 2
    python examples/dpt/04_end_to_end.py --no-labels
    python examples/dpt/04_end_to_end.py --skip-download
"""

import argparse
import sys
from pathlib import Path

from openai import APIError, AuthenticationError

from precisionai.agriemb import (
    build_client,
    download,
    embed_patch_tokens,
    run_dpt_evaluation,
    sample_cluster,
    verify_dataset,
)
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_MODEL, DEFAULT_REPO_ID
from precisionai.agriemb.dpt import DEFAULT_GRID_SIZE
from precisionai.agrieval.dpt import print_result
from precisionai.agristress.utils import AgriStressDataset

DEFAULT_CLUSTER = "L1"


def main() -> None:
    """Run the full verify-sample-tile-embed-evaluate pipeline for Dense Patch Tokens."""
    parser = argparse.ArgumentParser(
        description="End-to-end AgriStress-500 DPT demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    parser.add_argument("--skip-download", action="store_true", help="Never download; require a complete dataset")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help="L2 cluster to sample from")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of images to sample")
    parser.add_argument("--grid-height", type=int, default=DEFAULT_GRID_SIZE[0], help="Patch grid height (H), in tiles")
    parser.add_argument("--grid-width", type=int, default=DEFAULT_GRID_SIZE[1], help="Patch grid width (W), in tiles")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip ground-truth masks entirely for an unsupervised-metrics-only result",
    )
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
        missing = verify_dataset(dataset_dir)
        if missing:
            raise RuntimeError(
                f"{len(missing)} file(s) the manifests reference are still missing (e.g. {missing[:3]}), so the "
                f"download is incomplete and this run would fail on them. Re-run this script to resume it."
            )
    else:
        dataset_dir = Path(args.dataset_dir).resolve()
        print(f"[1/4] Dataset already complete at {dataset_dir}")
    dataset = AgriStressDataset(dataset_dir)

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
    print("[4/4] Dense Patch Token evaluation:")
    # Terminate agrieval's k-NN progress bar so it cannot collide with the summary.
    print(file=sys.stderr)
    print_result(result)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/01_download_dataset.py` first.")
    except AuthenticationError:
        sys.exit("Error: the PAI embedding API rejected PAI_EMBEDDINGS_API_KEY. Check .env against .env.example.")
    except APIError as err:
        sys.exit(f"Error: the PAI embedding API call failed ({err}). Wait a moment and re-run this script.")
    except (ConnectionError, RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
