# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Sample instance crops from one L2 cluster of the downloaded AgriStress-500 dataset.

Writes the selection to a JSON manifest that ``02_generate_embeddings.py``
reads.

Class diversity depends on the chosen cluster — most L2 clusters contain a
single dominant plant label, some contain several. The default is ``G2``,
which carries three distinct labels (two weeds and one crop); with a
single-label cluster every crop is a positive and the metrics saturate at
1.0000 without measuring anything. ``G2`` is the only three-label cluster;
``J1`` (12 images, 189 instance crops, ``Crop | Lentil`` and ``Weed | Grass``)
is the two-label alternative shown below.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/plant2plant/01_sample_cluster_instances.py
    python examples/plant2plant/01_sample_cluster_instances.py --cluster J1 --num-samples 6
"""

import argparse
import json
import sys
from pathlib import Path

from precisionai.agriemb import sample_cluster_instances
from precisionai.agriemb.config import DEFAULT_DATASET_DIR
from precisionai.agristress.utils import AgriStressDataset

DEFAULT_CLUSTER = "G2"
MANIFEST_PATH = Path("data/plant2plant_sample_manifest.json")


def main() -> None:
    """Sample a cluster's instance crops and write the selection manifest."""
    parser = argparse.ArgumentParser(
        description="Sample instance crops from one AgriStress-500 cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help="L2 cluster to sample instance crops from")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of parent images to draw crops from")
    parser.add_argument(
        "--max-instances-per-image", type=int, default=3, help="Max instance crops to keep per parent image"
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
    paths = dataset.resolve_paths(instances)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"dataset_dir": str(dataset_dir), "cluster": cluster, "instance_ids": instances}, indent=2)
    )

    print(f"Cluster sampled     : {cluster}")
    print(f"Source images       : {len(images)}")
    print(f"Instance crops      : {len(instances)}")
    for instance_id in instances:
        print(f"  {instance_id} -> {paths[instance_id]}")
    print(f"Manifest written to : {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/01_download_dataset.py` first.")
    except (RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
