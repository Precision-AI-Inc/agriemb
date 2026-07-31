# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run Dense Patch Token evaluation on the generated feature maps.

Reads the feature maps written by ``02_generate_features.py`` and evaluates
them with ``precisionai.agriemb.run_dpt_evaluation`` (which pairs them with
the AgriStress-500 dataset's ground-truth segmentation masks and class map,
then calls ``run_dpt_image_eval`` from ``agrieval``).

By default this validates each image's full-resolution ground-truth mask
for label-aware metrics, which costs on the order of a minute per image —
pass ``--no-labels`` to skip masks entirely for a near-instant,
unsupervised-metrics-only result.

Run from the project root (after ``02_generate_features.py``)::

    python examples/dpt/03_run_eval.py
    python examples/dpt/03_run_eval.py --no-labels
"""

import argparse
import json
from pathlib import Path

from precisionai.agriemb import run_dpt_evaluation
from precisionai.agrieval.dpt import print_result

FEATURES_PATH = Path("data/dpt_features.json")
EVAL_RESULT_PATH = Path("data/dpt_result.json")


def main() -> None:
    """Run the Dense Patch Token evaluation and print/save the result."""
    parser = argparse.ArgumentParser(description="Run DPT evaluation on generated feature maps")
    parser.add_argument("--features", default=str(FEATURES_PATH), help="Feature maps JSON from step 02")
    parser.add_argument("--output", default=str(EVAL_RESULT_PATH), help="Where to write the result JSON")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip ground-truth masks entirely for a near-instant, unsupervised-metrics-only result",
    )
    args = parser.parse_args()

    payload_source = json.loads(Path(args.features).read_text())
    dataset_dir = Path(payload_source["dataset_dir"])
    images = payload_source["images"]

    result = run_dpt_evaluation(dataset_dir, images, k_values=args.k_values, with_labels=not args.no_labels)
    print_result(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result written to {output_path}")


if __name__ == "__main__":
    main()
