# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven configuration for the AgriStress-500 pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # walks up from the cwd to find a `.env` file (see .env.example)

# Anchored to this repo's examples/ directory rather than the caller's cwd, so
# every script and notebook shares one downloaded copy of AgriStress-500
# (~12 GB) regardless of which directory it's launched from — a cwd-relative
# default would let each scenario subdirectory trigger its own fresh download.
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
DEFAULT_DATASET_DIR = _EXAMPLES_DIR / "data" / "agristress"
DEFAULT_REPO_ID = "precisionaiinc/AgriStress-500"
DEFAULT_MODEL = "pai-embedding"

# Small batch size for a single `embeddings.create` call. Full-resolution field
# images add up fast when base64-encoded inline in one JSON request, and large
# request bodies have been observed triggering connection resets against the
# hosted API — smaller, more numerous requests are more reliable than fewer,
# larger ones (well under the API's default `PAI_MAX_ITEMS` cap of 128 either way).
EMBEDDING_BATCH_SIZE = 2


def get_api_key() -> str:
    """Return the PAI embedding API key from the environment.

    Returns
    -------
    str
        The value of ``PAI_EMBEDDINGS_API_KEY``.

    Raises
    ------
    RuntimeError
        If the environment variable is unset (see ``.env.example``).
    """
    api_key = os.environ.get("PAI_EMBEDDINGS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PAI_EMBEDDINGS_API_KEY is not set — copy .env.example to .env and fill in your deployment's values."
        )
    return api_key


def get_base_url() -> str:
    """Return the PAI embedding API base URL from the environment.

    Returns
    -------
    str
        The value of ``PAI_EMBEDDINGS_BASE_URL``.

    Raises
    ------
    RuntimeError
        If the environment variable is unset (see ``.env.example``).
    """
    base_url = os.environ.get("PAI_EMBEDDINGS_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "PAI_EMBEDDINGS_BASE_URL is not set — copy .env.example to .env and fill in your deployment's values."
        )
    return base_url
