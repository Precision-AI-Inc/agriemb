# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dataset download and cluster sampling built on ``agristress-utils``."""

import random
from pathlib import Path

from precisionai.agriemb.config import DEFAULT_REPO_ID
from precisionai.agristress.utils import AgriStressDataset, download_dataset


def download(dataset_dir: str | Path, repo_id: str = DEFAULT_REPO_ID) -> Path:
    """Download the AgriStress-500 dataset.

    Parameters
    ----------
    dataset_dir : str or Path
        Local directory to download into.
    repo_id : str
        HuggingFace dataset repo id.

    Returns
    -------
    Path
        The resolved local dataset directory.
    """
    return download_dataset(dataset_dir, repo_id=repo_id)


def sample_cluster(
    dataset: AgriStressDataset,
    cluster: str | None = None,
    num_samples: int = 8,
    seed: int = 0,
) -> tuple[str, list[str]]:
    """Sample images from one L2 cluster of the dataset.

    Parameters
    ----------
    dataset : AgriStressDataset
        Loaded dataset to sample from.
    cluster : str or None
        L2 cluster id to sample from (e.g. ``"A1"``). A random cluster is
        chosen when omitted.
    num_samples : int
        Number of images to sample, capped to the cluster's size.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[str, list[str]]
        The chosen cluster id and the sampled image ids.

    Raises
    ------
    ValueError
        If ``cluster`` does not exist in the dataset.
    """
    rng = random.Random(seed)
    clusters = dataset.list_clusters()
    chosen = cluster or rng.choice(clusters)
    if chosen not in clusters:
        raise ValueError(f"Unknown cluster {chosen!r}; available clusters: {clusters}")

    image_ids = dataset.list_image_ids(cluster=chosen)
    sample = rng.sample(image_ids, k=min(num_samples, len(image_ids)))
    return chosen, sample
