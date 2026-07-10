# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image embedding generation via the PAI embedding API (https://embeddings.precision.ai/)."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from precisionai.agriemb.config import DEFAULT_MODEL, EMBEDDING_BATCH_SIZE, get_api_key, get_base_url


class _EmbeddingsResponse(Protocol):
    data: list[Any]  # each item has `.embedding: list[float]`


class _EmbeddingsResource(Protocol):
    def create(self, *, model: str, input: list[str]) -> _EmbeddingsResponse: ...


class EmbeddingsClient(Protocol):
    """Structural interface `embed_images` needs — satisfied by `openai.OpenAI`."""

    @property
    def embeddings(self) -> _EmbeddingsResource:
        """Embeddings resource exposing `create`."""
        ...


def build_client() -> OpenAI:
    """Build an OpenAI client configured for the PAI embedding API.

    Returns
    -------
    OpenAI
        Client authenticated with ``PAI_EMBEDDINGS_API_KEY`` and pointed at
        ``PAI_EMBEDDINGS_BASE_URL``.
    """
    return OpenAI(api_key=get_api_key(), base_url=get_base_url())


def image_to_data_uri(path: Path) -> str:
    """Encode an image file as a base64 ``data:`` URI for the embeddings API.

    Parameters
    ----------
    path : Path
        Path to a local image file.

    Returns
    -------
    str
        A ``data:<mime-type>;base64,<data>`` URI suitable as ``input`` to
        ``client.embeddings.create``.

    Raises
    ------
    ValueError
        If the file's extension does not map to a known image MIME type.
    """
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        raise ValueError(f"Could not determine an image MIME type for {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def embed_images(
    client: EmbeddingsClient,
    paths: dict[str, Path],
    model: str = DEFAULT_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> dict[str, list[float]]:
    """Embed every image in *paths* through the PAI embedding API.

    Parameters
    ----------
    client : EmbeddingsClient
        Client built by :func:`build_client` (or anything structurally
        compatible, e.g. a test double).
    paths : dict[str, Path]
        Image id → local file path.
    model : str
        Public model id to request.
    batch_size : int
        Maximum number of images sent per ``embeddings.create`` call.

    Returns
    -------
    dict[str, list[float]]
        Image id → raw (not yet normalized) embedding vector.
    """
    image_ids = list(paths)
    embeddings: dict[str, list[float]] = {}

    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        data_uris = [image_to_data_uri(paths[image_id]) for image_id in batch_ids]
        response = client.embeddings.create(model=model, input=data_uris)
        for image_id, item in zip(batch_ids, response.data, strict=True):
            embeddings[image_id] = item.embedding

    return embeddings
