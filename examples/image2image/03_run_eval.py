# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run Image→Image retrieval evaluation on the generated embeddings.

Reads the embeddings written by ``02_generate_embeddings.py`` and evaluates
them with ``precisionai.agriemb.run_i2i_evaluation`` (which wires
``AgriStressDataset.build_i2i_payload`` from ``agristress-utils`` into
``run_image2image_eval`` from ``agrieval``).

Run from the project root (after ``02_generate_embeddings.py``)::

    python examples/image2image/03_run_eval.py
"""

import argparse
import json
from pathlib import Path

from precisionai.agriemb import run_i2i_evaluation
from precisionai.agrieval.emb import print_result
from precisionai.agristress.utils import AgriStressDataset

EMBEDDINGS_PATH = Path("data/image2image_embeddings.json")
EVAL_RESULT_PATH = Path("data/image2image_result.json")


def main() -> None:
    """Run the Image→Image evaluation and print/save the result."""
    parser = argparse.ArgumentParser(description="Run Image2Image evaluation on generated embeddings")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="Embeddings JSON from step 02")
    parser.add_argument("--output", default=str(EVAL_RESULT_PATH), help="Where to write the result JSON")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="K cutoffs for KNN metrics")
    args = parser.parse_args()

    payload_source = json.loads(Path(args.embeddings).read_text())
    dataset_dir = Path(payload_source["dataset_dir"])
    embeddings = payload_source["embeddings"]

    dataset = AgriStressDataset(dataset_dir)
    result = run_i2i_evaluation(dataset, embeddings, k_values=args.k_values)
    print_result(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result written to {output_path}")


if __name__ == "__main__":
    main()
