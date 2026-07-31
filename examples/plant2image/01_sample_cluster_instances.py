# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Sample parent images and their instance crops from one L2 cluster.

Writes the selection to a JSON manifest that ``02_generate_embeddings.py``
reads.

The default cluster is one that is both well populated and usable. Only 80 of
AgriStress-500's 121 clusters work here at all: 27 own no instance crops (so
there would be nothing to retrieve) and 14 cannot be parsed by
``agristress-utils``, so a random cluster fails or degenerates about a third
of the time.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/plant2image/01_sample_cluster_instances.py
    python examples/plant2image/01_sample_cluster_instances.py --cluster C1 --num-samples 3
"""

import argparse
import json
import sys
from pathlib import Path

from precisionai.agriemb import sample_cluster_instances
from precisionai.agriemb.config import DEFAULT_DATASET_DIR
from precisionai.agristress.utils import AgriStressDataset

DEFAULT_CLUSTER = "A1"
MANIFEST_PATH = Path("data/plant2image_sample_manifest.json")


def main() -> None:
    """Sample a cluster's parent images and instance crops, and write the selection manifest."""
    parser = argparse.ArgumentParser(
        description="Sample parent images and instance crops from one L2 cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help="L2 cluster to sample from")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of parent images to sample")
    parser.add_argument(
        "--max-instances-per-image", type=int, default=4, help="Max instance crops to keep per parent image"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Where to write the sample manifest")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset = AgriStressDataset(dataset_dir)

    cluster, images, instances = sample_cluster_instances(
        dataset,
        cluster=args.cluster,
        num_samples=args.num_samples,
        max_instances_per_image=args.max_instances_per_image,
        seed=args.seed,
    )
    sample = images + instances
    paths = dataset.resolve_paths(sample)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {"dataset_dir": str(dataset_dir), "cluster": cluster, "image_ids": images, "instance_ids": instances},
            indent=2,
        )
    )

    print(f"Cluster sampled     : {cluster}")
    print(f"Parent images       : {len(images)}")
    print(f"Instance crops      : {len(instances)}")
    for item_id in sample:
        print(f"  {item_id} -> {paths[item_id]}")
    print(f"Manifest written to : {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/01_download_dataset.py` first.")
    except (RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
