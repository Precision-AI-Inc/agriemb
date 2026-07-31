# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Sample images from one L2 cluster of the downloaded AgriStress-500 dataset.

Writes the selection to a JSON manifest that ``02_generate_features.py``
reads. Kept small by default — every sampled image is tiled and every tile
embedded individually via the PAI embedding API in the next step, and
``03_run_eval.py``'s label-aware metrics validate each image's
full-resolution ground-truth mask (costing on the order of a minute per
image; see that script for a ``--no-labels`` escape hatch).

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/dpt/01_sample_cluster.py
    python examples/dpt/01_sample_cluster.py --cluster B1 --num-samples 2
"""

import argparse
import json
from pathlib import Path

from precisionai.agriemb import sample_cluster
from precisionai.agriemb.config import DEFAULT_DATASET_DIR
from precisionai.agristress.utils import AgriStressDataset

MANIFEST_PATH = Path("data/dpt_sample_manifest.json")


def main() -> None:
    """Sample a cluster's images and write the selection manifest."""
    parser = argparse.ArgumentParser(description="Sample images from one AgriStress-500 cluster")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--cluster", default=None, help="L2 cluster to sample from (default: random)")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Where to write the sample manifest")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset = AgriStressDataset(dataset_dir)

    cluster, sample = sample_cluster(dataset, cluster=args.cluster, num_samples=args.num_samples, seed=args.seed)
    paths = dataset.resolve_paths(sample)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"dataset_dir": str(dataset_dir), "cluster": cluster, "image_ids": sample}, indent=2)
    )

    print(f"Cluster sampled     : {cluster} ({len(dataset.list_image_ids(cluster=cluster))} images available)")
    print(f"Images selected     : {len(sample)}")
    for image_id in sample:
        print(f"  {image_id} -> {paths[image_id]}")
    print(f"Manifest written to : {manifest_path}")


if __name__ == "__main__":
    main()
