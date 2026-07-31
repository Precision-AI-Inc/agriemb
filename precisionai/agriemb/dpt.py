# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dense Patch Token (DPT) evaluation via ``agrieval``.

DPT scores the per-patch feature-map output of a dense vision backbone — the
spatial grid of patch embeddings a model produces before pooling into a
single image embedding — which is a different kind of artifact from the
pooled, whole-image vectors the hosted PAI embedding API returns directly.
No dense-feature backbone is wired up to that API, so :func:`tile_image` and
:func:`embed_patch_tokens` build a patch grid client-side instead: each image
is sliced into a grid of non-overlapping tiles, and every tile is embedded
individually through the real embedding API. Every value in the resulting
feature map is a genuine embedding from the hosted API — the tiling scheme
is the only synthetic part, not the features themselves. See
``examples/dpt/`` for runnable scripts.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.agriemb.config import DEFAULT_MODEL
from precisionai.agriemb.embeddings import EmbeddingsClient, embed_images
from precisionai.agrieval.dpt import run_dpt_image_eval

DEFAULT_GRID_SIZE = (8, 8)


def tile_image(path: Path, grid_size: tuple[int, int]) -> dict[str, Image.Image]:
    """Slice an image into a grid of non-overlapping tile crops.

    Parameters
    ----------
    path : Path
        Path to the source image file.
    grid_size : tuple[int, int]
        Patch grid ``(rows, cols)`` to slice the image into.

    Returns
    -------
    dict[str, PIL.Image.Image]
        ``"<row>,<col>"`` → cropped tile image, one entry per grid cell.
    """
    rows, cols = grid_size
    image = Image.open(path).convert("RGB")
    width, height = image.size

    tiles: dict[str, Image.Image] = {}
    for row in range(rows):
        y0, y1 = (row * height) // rows, ((row + 1) * height) // rows
        for col in range(cols):
            x0, x1 = (col * width) // cols, ((col + 1) * width) // cols
            tiles[f"{row},{col}"] = image.crop((x0, y0, x1, y1))
    return tiles


def embed_patch_tokens(
    client: EmbeddingsClient,
    paths: dict[str, Path],
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    model: str = DEFAULT_MODEL,
) -> dict[str, list[Any]]:
    """Build per-patch feature maps by tiling images and embedding every tile via the PAI embedding API.

    Every image in *paths* is sliced into a ``grid_size`` grid via
    :func:`tile_image`, every resulting tile is embedded individually
    through the real hosted embedding API, and the per-tile vectors are
    stacked back into a ``[P, H, W]`` feature map keyed by the source image.

    Parameters
    ----------
    client : EmbeddingsClient
        Client built by :func:`precisionai.agriemb.build_client`.
    paths : dict[str, Path]
        Image id (filename stem, e.g. ``"pai-OtvXFt3d"``, matching the
        AgriStress-500 mask filenames for label-aware metrics to align) →
        source image file path.
    grid_size : tuple[int, int]
        Patch grid ``(height, width)`` to tile each image into. Each tile is
        one ``embeddings.create`` call, so total API calls scale as
        ``len(paths) * grid_h * grid_w`` — keep this modest. Grids coarser
        than the 8x8 default are cheaper but leave every patch resolving to
        ``background`` on most clusters, which silently disables every
        label-aware metric in :func:`run_dpt_evaluation`.
    model : str
        Public model id to request.

    Returns
    -------
    dict[str, list[Any]]
        Image id → nested-list ``[P, H, W]`` feature map, JSON-serializable.

    Raises
    ------
    ValueError
        If *paths* is empty. An empty feature map would only fail later, and
        less legibly, inside :func:`run_dpt_evaluation`.
    """
    if not paths:
        raise ValueError("No images to embed: `paths` is empty, so there are no patch tokens to build.")

    grid_h, grid_w = grid_size
    tiles_by_image = {image_id: tile_image(path, grid_size) for image_id, path in paths.items()}

    all_tiles: dict[str, Path | Image.Image] = {
        f"{image_id}::{key}": tile for image_id, tiles in tiles_by_image.items() for key, tile in tiles.items()
    }
    tile_embeddings = embed_images(client, all_tiles, model=model, batch_size=1)
    embed_dim = len(next(iter(tile_embeddings.values())))

    images: dict[str, list[Any]] = {}
    for image_id, tiles in tiles_by_image.items():
        grid = np.zeros((embed_dim, grid_h, grid_w), dtype=np.float32)
        for key in tiles:
            row, col = (int(part) for part in key.split(","))
            grid[:, row, col] = tile_embeddings[f"{image_id}::{key}"]
        images[image_id] = grid.tolist()
    return images


def run_dpt_evaluation(
    dataset_dir: str | Path,
    images: dict[str, Any],
    k_values: list[int] | None = None,
    with_labels: bool = True,
) -> dict[str, Any]:
    """Evaluate patch-token feature maps with Dense Patch Token metrics.

    Pairs *images* with the AgriStress-500 dataset's ground-truth
    segmentation masks and class map for label-aware metrics, via
    :func:`precisionai.agrieval.dpt.run_dpt_image_eval`.

    Validating a mask requires decoding it at its full resolution (several
    megapixels for AgriStress-500) before it's downsampled to the patch
    grid, which costs roughly 15 s per image. Pass ``with_labels=False`` to
    skip masks entirely and get only the unsupervised patch-geometry
    metrics. Measured over two AgriStress-500 images on an 8x8 grid: 34 s
    label-aware, 4 s unsupervised-only.

    Parameters
    ----------
    dataset_dir : str or Path
        Local AgriStress-500 dataset directory — must contain ``masks/`` and
        ``class_map.json``.
    images : dict[str, Any]
        Image id (filename stem, e.g. ``"pai-OtvXFt3d"``) → ``[P, H, W]``
        patch-first feature map. Values may be nested lists or numpy arrays.
    k_values : list[int] or None
        K cutoffs for KNN metrics. Defaults to ``[5, 10, 20]``.
    with_labels : bool
        Whether to pair *images* with the dataset's ground-truth masks and
        class map for label-aware metrics. ``False`` skips masks entirely
        for a much faster, unsupervised-metrics-only result.

    Returns
    -------
    dict
        Fixed-schema result from ``run_dpt_image_eval``, ready for
        ``print_result`` or JSON serialization.
    """
    dataset_root = Path(dataset_dir)
    result = run_dpt_image_eval(
        images=images,
        k_values=k_values,
        masks_dir=dataset_root / "masks" if with_labels else None,
        classes_path=dataset_root / "class_map.json" if with_labels else None,
    )

    classes = result.get("classes") or []
    if with_labels and len(classes) < 2:
        warnings.warn(
            f"Every patch resolved to a single class ({classes or 'none'}), so silhouette, "
            f"calinski_harabasz, ari, nmi and pc1_auroc were all skipped and the k-NN purity "
            f"figures are trivially 1.0000. The patch grid is too coarse for these masks, or the "
            f"sampled images are near-uniform. Use a finer grid, or sample a cluster whose masks "
            f"carry more foreground.",
            stacklevel=2,
        )
    return result
