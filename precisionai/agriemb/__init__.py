# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""precisionai.agriemb — pipeline glue unifying the Precision AI embedding stack.

Connects three sibling projects into one download-sample-embed-evaluate
pipeline:

- ``precisionai.agriemb.dataset``    — download the AgriStress-500 dataset
  and sample a cluster (via ``agristress-utils``)
- ``precisionai.agriemb.embeddings`` — generate image embeddings through the
  hosted PAI embedding API (https://embeddings.precision.ai/, OpenAI SDK-compatible)
- ``precisionai.agriemb.evaluate``   — run Image→Image retrieval evaluation
  on the generated embeddings (via ``agrieval``)

See ``examples/`` for runnable scripts and a notebook built on this package.
"""

from precisionai.agriemb.dataset import download, sample_cluster
from precisionai.agriemb.embeddings import build_client, embed_images, image_to_data_uri
from precisionai.agriemb.evaluate import run_i2i_evaluation

__all__ = [
    "build_client",
    "download",
    "embed_images",
    "image_to_data_uri",
    "run_i2i_evaluation",
    "sample_cluster",
]
