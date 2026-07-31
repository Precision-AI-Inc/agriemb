# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image embedding generation via the PAI embedding API (https://embeddings.precision.ai/)."""

import base64
import io
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from PIL.Image import Image
from tqdm import tqdm

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


def image_array_to_data_uri(image: Image, image_format: str = "PNG") -> str:
    """Encode an in-memory image (e.g. a crop with no file of its own) as a base64 ``data:`` URI.

    Parameters
    ----------
    image : PIL.Image.Image
        In-memory image, such as a tile crop produced by
        :func:`precisionai.agriemb.dpt.tile_image`.
    image_format : str
        Format to encode with, per Pillow's ``Image.save`` ``format``
        argument (e.g. ``"PNG"``, ``"JPEG"``).

    Returns
    -------
    str
        A ``data:<mime-type>;base64,<data>`` URI suitable as ``input`` to
        ``client.embeddings.create``.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _any_image_to_data_uri(image: Path | Image) -> str:
    """Dispatch to :func:`image_to_data_uri` or :func:`image_array_to_data_uri` by type."""
    if isinstance(image, Path):
        return image_to_data_uri(image)
    return image_array_to_data_uri(image)


def embed_images(
    client: EmbeddingsClient,
    images: Mapping[str, Path | Image],
    model: str = DEFAULT_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Embed every image in *images* through the PAI embedding API.

    Parameters
    ----------
    client : EmbeddingsClient
        Client built by :func:`build_client` (or anything structurally
        compatible, e.g. a test double).
    images : Mapping[str, Path | PIL.Image.Image]
        Image id → local file path, or an in-memory image (e.g. a tile
        crop with no file of its own).
    model : str
        Public model id to request.
    batch_size : int
        Maximum number of images sent per ``embeddings.create`` call.
    progress : bool
        Show a progress bar over the batches. Each batch is a full-resolution
        round-trip taking several seconds, so a dozen images is a minute of
        otherwise silent waiting and the tiling path in
        :func:`precisionai.agriemb.dpt.embed_patch_tokens` is far longer.

    Returns
    -------
    dict[str, list[float]]
        Image id → raw (not yet normalized) embedding vector.
    """
    image_ids = list(images)
    embeddings: dict[str, list[float]] = {}

    batch_starts = range(0, len(image_ids), batch_size)
    for start in tqdm(batch_starts, desc="Embedding", unit="batch", disable=not progress):
        batch_ids = image_ids[start : start + batch_size]
        data_uris = [_any_image_to_data_uri(images[image_id]) for image_id in batch_ids]
        response = client.embeddings.create(model=model, input=data_uris)
        for image_id, item in zip(batch_ids, response.data, strict=True):
            embeddings[image_id] = item.embedding

    return embeddings
