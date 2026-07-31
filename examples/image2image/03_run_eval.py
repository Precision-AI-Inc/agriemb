# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run Image→Image retrieval evaluation on the generated embeddings.

Reads the embeddings written by ``02_generate_embeddings.py`` and evaluates
them with ``precisionai.agriemb.run_i2i_evaluation`` (which wires
``AgriStressDataset.build_i2i_payload`` from ``agristress-utils`` into
``run_image2image_eval`` from ``agrieval``).

The result JSON is written before the summary is printed, so a formatting
failure can never discard a completed evaluation.

Run from the project root (after ``02_generate_embeddings.py``)::

    python examples/image2image/03_run_eval.py
"""

import argparse
import json
import sys
from pathlib import Path

from precisionai.agriemb import print_evaluation, run_i2i_evaluation
from precisionai.agristress.utils import AgriStressDataset

EMBEDDINGS_PATH = Path("data/image2image_embeddings.json")
EVAL_RESULT_PATH = Path("data/image2image_result.json")


def main() -> None:
    """Run the Image→Image evaluation, save the result, then print it."""
    parser = argparse.ArgumentParser(
        description="Run Image2Image evaluation on generated embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="Embeddings JSON from step 02")
    parser.add_argument("--output", default=str(EVAL_RESULT_PATH), help="Where to write the result JSON")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    args = parser.parse_args()

    payload_source = json.loads(Path(args.embeddings).read_text())
    dataset_dir = Path(payload_source["dataset_dir"])
    embeddings = payload_source["embeddings"]

    dataset = AgriStressDataset(dataset_dir)
    result = run_i2i_evaluation(dataset, embeddings, k_values=args.k_values)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print_evaluation(result)
    print(f"\nFull result written to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/image2image/02_generate_embeddings.py` first.")
    except (RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
