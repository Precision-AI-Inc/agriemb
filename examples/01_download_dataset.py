# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Download the AgriStress-500 dataset used by the rest of the examples.

Run from the project root::

    python examples/01_download_dataset.py
    python examples/01_download_dataset.py --dataset-dir data/agristress
"""

import argparse

from precisionai.agriemb import download
from precisionai.agriemb.config import DEFAULT_DATASET_DIR, DEFAULT_REPO_ID
from precisionai.agristress.utils import AgriStressDataset


def main() -> None:
    """Download the dataset and print a summary of its cluster layout."""
    parser = argparse.ArgumentParser(description="Download the AgriStress-500 dataset")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local download directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id")
    args = parser.parse_args()

    dataset_dir = download(args.dataset_dir, repo_id=args.repo_id)
    dataset = AgriStressDataset(dataset_dir)
    clusters = dataset.list_clusters()

    print(f"Dataset downloaded to : {dataset_dir}")
    print(f"Clusters found        : {len(clusters)}")
    print(f"Images total          : {len(dataset.list_image_ids())}")
    print(f"First few clusters    : {clusters[:10]}")


if __name__ == "__main__":
    main()
