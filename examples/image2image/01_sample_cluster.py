# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Sample one L2 cluster of AgriStress-500 plus distractors from other clusters.

Writes the selection to a JSON manifest that ``02_generate_embeddings.py``
reads.

Image→Image treats same-cluster images as positives, so a sample drawn from a
single cluster has no negatives at all and every precision and nDCG figure
comes out at 1.0000 whatever the embeddings look like. Distractor images from
other clusters give the ranking something it can get wrong, which is what
makes the resulting numbers a measurement rather than arithmetic.

The default query cluster is a well-populated one. Most AgriStress-500
clusters are far too small to evaluate (104 of 121 hold fewer than 8 images,
38 hold exactly one), so a random cluster would usually produce a saturated,
meaningless result.

Run from the project root (after ``examples/01_download_dataset.py``)::

    python examples/image2image/01_sample_cluster.py
    python examples/image2image/01_sample_cluster.py --cluster C1 --num-samples 12
"""

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

from precisionai.agriemb import sample_cluster_with_distractors
from precisionai.agriemb.config import DEFAULT_DATASET_DIR
from precisionai.agristress.utils import AgriStressDataset

DEFAULT_CLUSTER = "A1"
MANIFEST_PATH = Path("data/image2image_sample_manifest.json")


def main() -> None:
    """Sample a query cluster plus cross-cluster distractors and write the selection manifest."""
    parser = argparse.ArgumentParser(
        description="Sample one AgriStress-500 cluster plus cross-cluster distractors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Local dataset directory")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help="L2 cluster to use as the query cluster")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of query-cluster images to sample")
    parser.add_argument(
        "--num-distractor-clusters", type=int, default=6, help="Number of other clusters to draw distractors from"
    )
    parser.add_argument(
        "--num-distractor-samples", type=int, default=2, help="Number of images to draw from each distractor cluster"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Where to write the sample manifest")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset = AgriStressDataset(dataset_dir)

    cluster, sample, distractors = sample_cluster_with_distractors(
        dataset,
        cluster=args.cluster,
        num_samples=args.num_samples,
        num_distractor_clusters=args.num_distractor_clusters,
        num_distractor_samples=args.num_distractor_samples,
        seed=args.seed,
    )
    paths = dataset.resolve_paths(sample + distractors)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "cluster": cluster,
                "image_ids": sample,
                "distractor_ids": distractors,
            },
            indent=2,
        )
    )

    distractor_clusters = sorted(
        {PurePosixPath(distractor_id).parent.name or distractor_id for distractor_id in distractors}
    )
    print(f"Query cluster       : {cluster} ({len(dataset.list_image_ids(cluster=cluster))} images available)")
    print(f"Query images        : {len(sample)}")
    for image_id in sample:
        print(f"  {image_id} -> {paths[image_id]}")
    print(f"Distractor images   : {len(distractors)} from {len(distractor_clusters)} clusters {distractor_clusters}")
    for image_id in distractors:
        print(f"  {image_id} -> {paths[image_id]}")
    print(f"Manifest written to : {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/01_download_dataset.py` first.")
    except (RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
