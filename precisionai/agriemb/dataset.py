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


def sample_cluster_instances(
    dataset: AgriStressDataset,
    cluster: str | None = None,
    num_samples: int = 4,
    max_instances_per_image: int | None = 4,
    seed: int = 0,
) -> tuple[str, list[str], list[str]]:
    """Sample parent images from one L2 cluster and collect their instance crops.

    Built on :func:`sample_cluster` for the parent images, then reads each
    image's instance crops from ``plant2image.json`` via
    :meth:`AgriStressDataset.image_metadata`. Used to assemble the combined
    parent-plus-instance embedding sets required by the Plant→Image and
    Plant→Plant evaluation scenarios.

    Parameters
    ----------
    dataset : AgriStressDataset
        Loaded dataset to sample from.
    cluster : str or None
        L2 cluster id to sample from (e.g. ``"A1"``). A random cluster is
        chosen when omitted. Plant class diversity varies by cluster — some
        clusters contain a single crop/weed label, others several.
    num_samples : int
        Number of parent images to sample, capped to the cluster's size.
    max_instances_per_image : int or None
        Maximum instance crops to keep per sampled parent image, capped to
        the image's instance count. ``None`` keeps every instance crop,
        which can be dozens per image.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[str, list[str], list[str]]
        The chosen cluster id, the sampled parent image ids, and the
        (possibly capped) instance crop ids extracted from those images.

    Raises
    ------
    ValueError
        If ``cluster`` does not exist in the dataset.
    """
    rng = random.Random(seed)
    chosen, image_ids = sample_cluster(dataset, cluster=cluster, num_samples=num_samples, seed=seed)
    records = dataset.image_metadata(image_ids)

    instance_ids: list[str] = []
    for record in records:
        available = record.instances
        if max_instances_per_image is not None and len(available) > max_instances_per_image:
            available = rng.sample(available, k=max_instances_per_image)
        instance_ids.extend(available)

    return chosen, image_ids, instance_ids
