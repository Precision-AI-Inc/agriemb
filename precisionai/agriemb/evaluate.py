# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image→Image retrieval evaluation via ``agrieval``."""

from precisionai.agrieval.emb import MetadataGroup, run_image2image_eval
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
