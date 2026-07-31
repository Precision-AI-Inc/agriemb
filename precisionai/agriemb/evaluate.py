# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image→Image, Plant→Image, and Plant→Plant retrieval evaluation via ``agrieval``."""

from precisionai.agrieval.emb import (
    MetadataGroup,
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)
from precisionai.agristress.utils import AgriStressDataset


def run_i2i_evaluation(
    dataset: AgriStressDataset,
    embeddings: dict[str, list[float]],
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate normalized embeddings with Image→Image retrieval metrics.

    Assembles the explicit-positive metadata groups from the dataset's
    ``image2image.json`` via :meth:`AgriStressDataset.build_i2i_payload`, then
    evaluates them with :func:`precisionai.agrieval.emb.run_image2image_eval`.

    Parameters
    ----------
    dataset : AgriStressDataset
        Dataset the embeddings were sampled from.
    embeddings : dict[str, list[float]]
        Image id → L2-normalized embedding vector.
    k_values : list[int] or None
        K cutoffs for KNN metrics. Defaults to ``[5, 10, 20]``.

    Returns
    -------
    dict
        Fixed-schema result from ``run_image2image_eval``, ready for
        ``print_result`` or JSON serialization.
    """
    payload = dataset.build_i2i_payload(embeddings, dataset_root="images", k_values=k_values, sample_pairs=None)

    metadata = (
        {name: MetadataGroup.model_validate(group) for name, group in payload["metadata"].items()}
        if payload["metadata"]
        else None
    )

    return run_image2image_eval(
        image_embeddings=payload["embeddings"],
        k_values=payload["k_values"],
        dataset_root=payload["dataset_root"],
        sample_pairs=payload["sample_pairs"],
        metadata=metadata,
    )


def run_p2i_evaluation(
    dataset: AgriStressDataset,
    embeddings: dict[str, list[float]],
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate normalized embeddings with Plant→Image retrieval metrics.

    Assembles the parent-image-to-instance-crop groups from
    ``plant2image.json`` via :meth:`AgriStressDataset.build_p2i_payload`, then
    evaluates them with :func:`precisionai.agrieval.emb.run_plant2image_eval`.

    Parameters
    ----------
    dataset : AgriStressDataset
        Dataset the embeddings were sampled from.
    embeddings : dict[str, list[float]]
        Parent image id and instance crop id → L2-normalized embedding
        vector. Must include both the parent images and their instance crops.
    k_values : list[int] or None
        K cutoffs for KNN metrics. Defaults to ``[5, 10, 20]``.

    Returns
    -------
    dict
        Fixed-schema result from ``run_plant2image_eval``, ready for
        ``print_result`` or JSON serialization.
    """
    payload = dataset.build_p2i_payload(embeddings, dataset_root="images", k_values=k_values, sample_pairs=None)

    return run_plant2image_eval(
        embeddings=payload["embeddings"],
        instance_to_image=payload["instance_to_image"],
        k_values=payload["k_values"],
        dataset_root=payload["dataset_root"],
        sample_pairs=payload["sample_pairs"],
    )


def run_p2p_evaluation(
    dataset: AgriStressDataset,
    embeddings: dict[str, list[float]],
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate normalized embeddings with Plant→Plant retrieval metrics.

    Assembles the instance crop class labels from ``plant2plant.json`` via
    :meth:`AgriStressDataset.build_p2p_payload`, then evaluates them with
    :func:`precisionai.agrieval.emb.run_plant2plant_eval`.

    Parameters
    ----------
    dataset : AgriStressDataset
        Dataset the embeddings were sampled from.
    embeddings : dict[str, list[float]]
        Instance crop id → L2-normalized embedding vector.
    k_values : list[int] or None
        K cutoffs for KNN metrics. Defaults to ``[5, 10, 20]``.

    Returns
    -------
    dict
        Fixed-schema result from ``run_plant2plant_eval``, ready for
        ``print_result`` or JSON serialization.
    """
    payload = dataset.build_p2p_payload(embeddings, k_values=k_values, sample_pairs=None)

    return run_plant2plant_eval(
        embeddings=payload["embeddings"],
        instance_labels=payload["instance_labels"],
        k_values=payload["k_values"],
        sample_pairs=payload["sample_pairs"],
    )
