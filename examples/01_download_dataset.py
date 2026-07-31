# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Download the AgriStress-500 dataset used by the rest of the examples.

Shared prerequisite for all four scenarios under ``examples/``:
``image2image/``, ``plant2image/``, ``plant2plant/``, and ``dpt/`` (which
needs the ground-truth masks as well as the images). Each scenario directory
has its own numbered ``sample`` → ``generate`` → ``run_eval`` scripts plus a
bundled ``04_end_to_end.py``.

The dataset is ~12 GB across 4950 files and takes 20+ minutes to fetch on a
fast connection. HuggingFace rate-limits partway through often enough that
failed attempts are retried automatically; if the download still gives up,
re-run this script and it resumes from whatever is already on disk.

Run from the project root::

    python examples/01_download_dataset.py
    python examples/01_download_dataset.py --dataset-dir data/agristress
"""

import argparse
import sys

from precisionai.agriemb import download, verify_dataset
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_REPO_ID
from precisionai.agristress.utils import AgriStressDataset


def main() -> None:
    """Download the dataset and print a summary of its cluster layout."""
    parser = argparse.ArgumentParser(
        description="Download the AgriStress-500 dataset (~12 GB, 4950 files, 20+ minutes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local download directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    args = parser.parse_args()

    print(f"Downloading {args.repo_id} (~12 GB, 4950 files) into {args.dataset_dir}")
    print("This takes 20+ minutes on a fast connection, and resumes if it is interrupted.")

    dataset_dir = download(args.dataset_dir, repo_id=args.repo_id)

    missing = verify_dataset(dataset_dir)
    if missing:
        raise RuntimeError(
            f"{len(missing)} file(s) the manifests reference are still missing (e.g. {missing[:3]}), so the "
            f"download is incomplete and the scenario scripts would fail on them. Re-run this script to "
            f"resume it."
        )

    dataset = AgriStressDataset(dataset_dir)
    clusters = dataset.list_clusters()

    print(f"Dataset downloaded to : {dataset_dir}")
    print(f"Clusters found        : {len(clusters)}")
    print(f"Images total          : {len(dataset.list_image_ids())}")
    print(f"Files still missing   : {len(missing)}")
    print(f"First few clusters    : {clusters[:10]}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: the local dataset copy is incomplete ({err}). Re-run this script; the download resumes.")
    except (ConnectionError, RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
