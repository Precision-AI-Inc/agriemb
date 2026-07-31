# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""precisionai.agriemb — pipeline glue unifying the Precision AI embedding stack.

Connects three sibling projects into one download-sample-embed-evaluate
pipeline:

- ``precisionai.agriemb.dataset``    — download and verify the AgriStress-500
  dataset, and sample a cluster, a cluster plus cross-cluster distractors, or
  a cluster's instance crops (via ``agristress-utils``)
- ``precisionai.agriemb.embeddings`` — generate image embeddings through the
  hosted PAI embedding API (https://embeddings.precision.ai/, OpenAI SDK-compatible)
- ``precisionai.agriemb.evaluate``   — run Image→Image, Plant→Image, and
  Plant→Plant retrieval evaluation on the generated embeddings (via ``agrieval``)
- ``precisionai.agriemb.dpt``        — build per-patch feature maps by tiling
  images and embedding every tile through the hosted PAI embedding API (no
  dense-feature backbone is wired to it directly), then run Dense Patch
  Token evaluation on them (via ``agrieval``)

See ``examples/`` for runnable scripts and a notebook per scenario, built on
this package.
"""

from precisionai.agriemb.dataset import (
    download,
    sample_cluster,
    sample_cluster_instances,
    sample_cluster_with_distractors,
    verify_dataset,
)
from precisionai.agriemb.dpt import DEFAULT_GRID_SIZE, embed_patch_tokens, run_dpt_evaluation, tile_image
from precisionai.agriemb.embeddings import build_client, embed_images, image_array_to_data_uri, image_to_data_uri
from precisionai.agriemb.evaluate import (
    print_evaluation,
    run_i2i_evaluation,
    run_p2i_evaluation,
    run_p2p_evaluation,
)

__all__ = [
    "DEFAULT_GRID_SIZE",
    "build_client",
    "download",
    "embed_images",
    "embed_patch_tokens",
    "image_array_to_data_uri",
    "image_to_data_uri",
    "print_evaluation",
    "run_dpt_evaluation",
    "run_i2i_evaluation",
    "run_p2i_evaluation",
    "run_p2p_evaluation",
    "sample_cluster",
    "sample_cluster_instances",
    "sample_cluster_with_distractors",
    "tile_image",
    "verify_dataset",
]
