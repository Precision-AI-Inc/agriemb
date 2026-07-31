# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image→Image, Plant→Image, and Plant→Plant retrieval evaluation via ``agrieval``."""

import sys
import warnings

from precisionai.agrieval.emb import (
    MetadataGroup,
    print_result,
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)
from precisionai.agristress.utils import AgriStressDataset

# Below this, every retrieval metric is either undefined or trivially 1.0000.
MIN_EVALUATION_ITEMS = 3


def _check_corpus_size(embeddings: dict[str, list[float]], scenario: str) -> None:
    """Reject a corpus too small for any retrieval metric to carry information.

    Parameters
    ----------
    embeddings : dict[str, list[float]]
        The embeddings about to be evaluated.
    scenario : str
        Scenario name used in the error message.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_EVALUATION_ITEMS` embeddings were supplied.
    """
    if len(embeddings) < MIN_EVALUATION_ITEMS:
        raise ValueError(
            f"{scenario} needs at least {MIN_EVALUATION_ITEMS} embeddings to produce a meaningful "
            f"ranking, but {len(embeddings)} were supplied. Sample more images, or sample a larger "
            f"cluster (see `precisionai.agriemb.sample_cluster`)."
        )


def _warn_on_degenerate_result(
    result: dict,
    requested_k: list[int] | None,
    classes_define_positives: bool = True,
) -> None:
    """Warn when the corpus was too small for the requested metrics to mean anything.

    ``agrieval`` silently clamps every requested k to the candidate pool and
    reports the clamped values back as ``k_values``, so a request for
    ``[5, 10]`` over two items becomes a k=1 evaluation whose scores are all
    1.0000 with nothing anywhere saying so.

    Parameters
    ----------
    result : dict
        Fixed-schema evaluation result.
    requested_k : list[int] or None
        The k values the caller asked for, or ``None`` for ``agrieval``'s
        defaults.
    classes_define_positives : bool
        Whether ``result["classes"]`` is what makes a candidate relevant.
        True for Image→Image (same cluster) and Plant→Plant (same crop/weed
        label); False for Plant→Image, where relevance comes from the
        parent-image-to-instance-crop groups instead and a single class is
        perfectly normal.
    """
    n_items = result["n_items"]
    effective_k = result["k_values"]

    if requested_k is not None and sorted(effective_k) != sorted(requested_k):
        warnings.warn(
            f"Requested k values {sorted(requested_k)} were clamped to {sorted(effective_k)}: only "
            f"{n_items} items were evaluated, so larger cutoffs have no candidates to rank.",
            stacklevel=3,
        )

    if effective_k and n_items - 1 <= max(effective_k):
        warnings.warn(
            f"Only {n_items} items were evaluated at k={sorted(effective_k)}. Every candidate fits "
            f"inside the largest cutoff, so precision and nDCG saturate at 1.0000 by construction "
            f"and measure nothing about embedding quality.",
            stacklevel=3,
        )

    classes = result.get("classes") or []
    if classes_define_positives and len(classes) < 2:
        warnings.warn(
            f"All {n_items} evaluated items share one class ({classes or 'none'}), so every "
            f"candidate is a positive and there is nothing for the ranking to get wrong. Precision "
            f"and nDCG are 1.0000 by construction. Add items from other clusters or classes.",
            stacklevel=3,
        )


def print_evaluation(result: dict) -> None:
    """Print an evaluation result, reporting undefined metrics as ``n/a``.

    ``agrieval``'s ``print_result`` formats every global metric with
    ``:.4f``, which raises ``TypeError`` on the ``None`` it legitimately
    reports for a metric a degenerate sample leaves undefined (``alignment``
    with no positive pairs, for instance) — losing the whole summary to a
    traceback. Undefined metrics are pulled out and listed as ``n/a``
    instead, so the rest of the report still prints.

    Also terminates the ``k-NN graph`` progress bar ``agrieval`` leaves open
    on stderr, which would otherwise collide with the first summary line.

    Parameters
    ----------
    result : dict
        Fixed-schema result from any of the ``run_*_evaluation`` functions.
    """
    print(file=sys.stderr)

    global_metrics = result["global_metrics"]
    undefined = sorted(name for name, value in global_metrics.items() if value is None)

    printable = dict(result)
    printable["global_metrics"] = {name: value for name, value in global_metrics.items() if name not in undefined}
    print_result(printable)

    for name in undefined:
        print(f"  {name:<19}: n/a  (undefined for this sample)")


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
    _check_corpus_size(embeddings, "Image→Image evaluation")
    payload = dataset.build_i2i_payload(embeddings, dataset_root="images", k_values=k_values, sample_pairs=None)

    metadata = (
        {name: MetadataGroup.model_validate(group) for name, group in payload["metadata"].items()}
        if payload["metadata"]
        else None
    )

    result = run_image2image_eval(
        image_embeddings=payload["embeddings"],
        k_values=payload["k_values"],
        dataset_root=payload["dataset_root"],
        sample_pairs=payload["sample_pairs"],
        metadata=metadata,
    )
    _warn_on_degenerate_result(result, k_values)
    return result


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
    _check_corpus_size(embeddings, "Plant→Image evaluation")
    payload = dataset.build_p2i_payload(embeddings, dataset_root="images", k_values=k_values, sample_pairs=None)

    result = run_plant2image_eval(
        embeddings=payload["embeddings"],
        instance_to_image=payload["instance_to_image"],
        k_values=payload["k_values"],
        dataset_root=payload["dataset_root"],
        sample_pairs=payload["sample_pairs"],
    )
    # Plant→Image relevance is parent↔instance, not the path-derived class label.
    _warn_on_degenerate_result(result, k_values, classes_define_positives=False)
    return result


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
    _check_corpus_size(embeddings, "Plant→Plant evaluation")
    payload = dataset.build_p2p_payload(embeddings, k_values=k_values, sample_pairs=None)

    result = run_plant2plant_eval(
        embeddings=payload["embeddings"],
        instance_labels=payload["instance_labels"],
        k_values=payload["k_values"],
        sample_pairs=payload["sample_pairs"],
    )
    _warn_on_degenerate_result(result, k_values)
    return result
