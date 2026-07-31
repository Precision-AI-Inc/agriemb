# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run Dense Patch Token evaluation on the generated feature maps.

Reads the feature maps written by ``02_generate_features.py`` and evaluates
them with ``precisionai.agriemb.run_dpt_evaluation`` (which pairs them with
the AgriStress-500 dataset's ground-truth segmentation masks and class map,
then calls ``run_dpt_image_eval`` from ``agrieval``).

By default this validates each image's full-resolution ground-truth mask for
label-aware metrics, which costs roughly 15 s per image; pass ``--no-labels``
to skip masks entirely for an unsupervised-metrics-only result. Measured on
the two-image default: 34 s label-aware, 4 s with ``--no-labels``.

The result JSON is written before the summary is printed, so a formatting
failure can never discard a completed evaluation.

Run from the project root (after ``02_generate_features.py``)::

    python examples/dpt/03_run_eval.py
    python examples/dpt/03_run_eval.py --no-labels
"""

import argparse
import json
import sys
from pathlib import Path

from precisionai.agriemb import run_dpt_evaluation
from precisionai.agrieval.dpt import print_result

FEATURES_PATH = Path("data/dpt_features.json")
EVAL_RESULT_PATH = Path("data/dpt_result.json")


def main() -> None:
    """Run the Dense Patch Token evaluation, save the result, then print it."""
    parser = argparse.ArgumentParser(
        description="Run DPT evaluation on generated feature maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", default=str(FEATURES_PATH), help="Feature maps JSON from step 02")
    parser.add_argument("--output", default=str(EVAL_RESULT_PATH), help="Where to write the result JSON")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip ground-truth masks entirely for an unsupervised-metrics-only result",
    )
    args = parser.parse_args()

    payload_source = json.loads(Path(args.features).read_text())
    dataset_dir = Path(payload_source["dataset_dir"])
    images = payload_source["images"]

    result = run_dpt_evaluation(dataset_dir, images, k_values=args.k_values, with_labels=not args.no_labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    # Terminate agrieval's k-NN progress bar so it cannot collide with the summary.
    print(file=sys.stderr)
    print_result(result)
    print(f"\nFull result written to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/dpt/02_generate_features.py` first.")
    except (RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
