# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Build patch-token feature maps by tiling sampled images and embedding every tile.

Reads the manifest produced by ``01_sample_cluster.py``, slices each sampled
image into a grid of tiles, embeds every tile individually through the
OpenAI-compatible PAI embedding API (https://embeddings.precision.ai/), and
stacks the resulting real per-tile vectors into a ``[P, H, W]`` feature map
per image. No dense-feature backbone is wired up to the hosted API (it
returns one pooled vector per image, not a spatial patch grid), so this
tiling scheme is what builds the patch grid — every value in it is a
genuine embedding from the real API.

Each tile is a separate ``embeddings.create`` call, so total API calls scale
as ``num_samples * grid_height * grid_width``: 2 x 8 x 8 = 128 by default.
Grids coarser than 8x8 are cheaper but leave every patch resolving to
``background`` on most clusters, which silently disables every label-aware
metric in ``03_run_eval.py``.

Requires ``PAI_EMBEDDINGS_API_KEY`` and ``PAI_EMBEDDINGS_BASE_URL`` — copy
``.env.example`` to ``.env`` and fill them in.

Run from the project root (after ``01_sample_cluster.py``)::

    python examples/dpt/02_generate_features.py
"""

import argparse
import json
import sys
from pathlib import Path

from openai import AuthenticationError

from precisionai.agriemb import build_client, embed_patch_tokens
from precisionai.agriemb.config import DEFAULT_MODEL
from precisionai.agriemb.dpt import DEFAULT_GRID_SIZE
from precisionai.agristress.utils import AgriStressDataset

MANIFEST_PATH = Path("data/dpt_sample_manifest.json")
FEATURES_PATH = Path("data/dpt_features.json")


def main() -> None:
    """Tile sampled images, embed every tile, and write the feature maps to a JSON file."""
    parser = argparse.ArgumentParser(
        description="Build patch-token feature maps for sampled images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Sample manifest from step 01")
    parser.add_argument("--output", default=str(FEATURES_PATH), help="Where to write the feature maps JSON")
    parser.add_argument("--grid-height", type=int, default=DEFAULT_GRID_SIZE[0], help="Patch grid height (H), in tiles")
    parser.add_argument("--grid-width", type=int, default=DEFAULT_GRID_SIZE[1], help="Patch grid width (W), in tiles")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Public model id to request")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    dataset_dir = Path(manifest["dataset_dir"])
    image_ids = manifest["image_ids"]

    dataset = AgriStressDataset(dataset_dir)
    paths = dataset.resolve_paths(image_ids)
    stems = {Path(image_id).stem: path for image_id, path in paths.items()}

    client = build_client()
    images = embed_patch_tokens(client, stems, grid_size=(args.grid_height, args.grid_width), model=args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"dataset_dir": str(dataset_dir), "images": images}))

    n_tiles = len(stems) * args.grid_height * args.grid_width
    print(
        f"Embedded {n_tiles} tiles across {len(images)} images into "
        f"[H={args.grid_height}, W={args.grid_width}] feature maps, written to {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}\nRun `python examples/dpt/01_sample_cluster.py` first.")
    except AuthenticationError:
        sys.exit("Error: the PAI embedding API rejected PAI_EMBEDDINGS_API_KEY. Check .env against .env.example.")
    except (ConnectionError, RuntimeError, ValueError) as err:
        sys.exit(f"Error: {err}")
