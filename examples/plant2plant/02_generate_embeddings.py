# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Embed sampled instance crops via the PAI embedding API.

Reads the manifest produced by ``01_sample_cluster_instances.py``, embeds
each instance crop through the OpenAI-compatible PAI embedding API
(https://embeddings.precision.ai/), L2-normalizes and validates the resulting
vectors, and writes them to a JSON file for ``03_run_eval.py``.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``01_sample_cluster_instances.py``)::

    python examples/plant2plant/02_generate_embeddings.py
"""

import argparse
import json
from pathlib import Path

from precisionai.agriemb import build_client, embed_images
from precisionai.agriemb.config import DEFAULT_MODEL
from precisionai.agristress.utils import AgriStressDataset, normalize_embeddings, validate_embeddings

MANIFEST_PATH = Path("data/plant2plant_sample_manifest.json")
EMBEDDINGS_PATH = Path("data/plant2plant_embeddings.json")


def main() -> None:
    """Embed the sampled instance crops and write the normalized embeddings."""
    parser = argparse.ArgumentParser(description="Embed sampled instance crops via the PAI embedding API")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Sample manifest from step 01")
    parser.add_argument("--output", default=str(EMBEDDINGS_PATH), help="Where to write the embeddings JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    dataset_dir = Path(manifest["dataset_dir"])
    instance_ids = manifest["instance_ids"]

    dataset = AgriStressDataset(dataset_dir)
    paths = dataset.resolve_paths(instance_ids)

    client = build_client()
    raw_embeddings = embed_images(client, paths, model=args.model)
    embeddings = normalize_embeddings(raw_embeddings)
    validate_embeddings(embeddings)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"dataset_dir": str(dataset_dir), "embeddings": embeddings}))

    print(f"Wrote {len(embeddings)} L2-normalized embeddings to {output_path}")


if __name__ == "__main__":
    main()
