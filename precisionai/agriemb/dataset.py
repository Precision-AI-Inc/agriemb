# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dataset download and cluster sampling built on ``agristress-utils``."""

import glob
import random
import sys
import time
import warnings
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from precisionai.agriemb.config import DEFAULT_REPO_ID
from precisionai.agristress.utils import AgriStressDataset, download_dataset

DOWNLOAD_MAX_ATTEMPTS = 4
DOWNLOAD_INITIAL_BACKOFF_SECONDS = 5.0

RATE_LIMITED_MESSAGE = "Download interrupted (rate-limited). It is resumable - re-run this script."

# Non-5xx HTTP statuses a later attempt could still succeed on. Everything
# else with a status (401, 403, 404, ...) fails the same way however often it
# is retried.
_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429})

# Files the dataset root must contain before any cluster can be enumerated.
_REQUIRED_MANIFESTS = ("image2image.json", "plant2image.json", "plant2plant.json", "class_map.json")

# How many alternative cluster ids to name when an error message suggests one.
_SUGGESTION_COUNT = 8


def _attempt_download(dataset_dir: str | Path, repo_id: str) -> Path | OSError:
    """Run one download attempt, returning the error rather than raising it.

    Rate limits, connection resets, timeouts and HTTP errors from the Hub all
    subclass ``OSError``, as do permission and disk-space failures;
    :func:`_is_transient` tells them apart. Returning the error keeps
    :func:`download`'s retry loop free of a ``try``/``except`` in its body.
    """
    try:
        return download_dataset(dataset_dir, repo_id=repo_id)
    except OSError as err:
        return err


def _is_transient(error: OSError) -> bool:
    """Report whether a failed download attempt could succeed if it were retried.

    ``huggingface_hub``, ``httpx`` and the xet client raise unrelated
    exception classes that share only ``OSError`` as an ancestor, so the
    decision is made on the HTTP status when the error carries a response and
    on the exception type otherwise. A gated or missing repo, a rejected
    token, a read-only directory or a full disk is never retried: it would
    fail identically on every attempt and reporting it as rate limiting sends
    the user off to re-run a command that cannot work.

    Parameters
    ----------
    error : OSError
        The error a download attempt failed with.

    Returns
    -------
    bool
        True for rate limits, server-side failures and connection or timeout
        errors; False for everything else.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_HTTP_STATUS or status >= 500
    return isinstance(error, ConnectionError | TimeoutError)


def download(
    dataset_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS,
) -> Path:
    """Download the AgriStress-500 dataset, retrying transient network failures.

    The dataset is ~12 GB across 4950 files and takes 20+ minutes to fetch on
    a fast connection. HuggingFace rate-limits (HTTP 429) partway through
    often enough that a single attempt is unreliable, so transient failures
    are retried with exponential backoff. Every attempt resumes from whatever
    is already on disk, so re-running after a failure never restarts from
    zero. Failures a retry cannot fix (a missing or gated repo, a rejected
    token, a read-only directory, a full disk) are reported immediately
    instead, since backing off and re-running only delays the same error.

    Parameters
    ----------
    dataset_dir : str or Path
        Local directory to download into.
    repo_id : str
        HuggingFace dataset repo id.
    max_attempts : int
        How many times to attempt the download before giving up.

    Returns
    -------
    Path
        The resolved local dataset directory.

    Raises
    ------
    ConnectionError
        If every attempt failed with a transient network error. Whatever was
        already downloaded stays on disk, so re-running resumes from there.
    RuntimeError
        If an attempt failed with an error a retry cannot fix. The underlying
        error is named in the message and chained as the cause.
    """
    last_error: OSError | None = None
    for attempt in range(1, max_attempts + 1):
        outcome = _attempt_download(dataset_dir, repo_id)
        if not isinstance(outcome, OSError):
            return outcome
        if not _is_transient(outcome):
            raise RuntimeError(
                f"Downloading {repo_id} into {dataset_dir} failed for a reason retrying cannot fix, so it "
                f"was not retried: {outcome}"
            ) from outcome

        last_error = outcome
        if attempt < max_attempts:
            backoff = DOWNLOAD_INITIAL_BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"Download attempt {attempt}/{max_attempts} failed ({outcome}). Retrying in {backoff:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(backoff)

    raise ConnectionError(f"{RATE_LIMITED_MESSAGE} Last error: {last_error}") from last_error


def _missing_masks(root: Path, image_ids: list[str]) -> list[str]:
    """Return the dataset-relative mask paths no file on disk answers to.

    Masks mirror the image tree (``images/A1/pai-x.png`` ->
    ``masks/A1/pai-x.*``) but nothing guarantees they share the image's file
    extension, so each one is resolved by stem the way ``agrieval`` resolves
    its own ground-truth masks. Substituting the image extension instead
    would report every mask in a complete dataset as missing.

    Parameters
    ----------
    root : Path
        Local dataset directory.
    image_ids : list[str]
        Dataset-relative image ids whose masks to look for.

    Returns
    -------
    list[str]
        The expected mask paths that resolved to nothing, named after their
        image so the message points at a real dataset entry.
    """
    missing: list[str] = []
    for image_id in image_ids:
        relative = PurePosixPath(image_id.split("/", 1)[1])
        candidates = (root / "masks" / relative.parent).glob(f"{glob.escape(relative.stem)}.*")
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(f"masks/{relative}")
    return missing


def verify_dataset(dataset_dir: str | Path) -> list[str]:
    """Report which files the AgriStress-500 manifests reference but that are absent on disk.

    Lets callers skip a completed download instead of re-walking all 4950
    files, and catches the half-downloaded state a rate-limited run leaves
    behind (see :func:`download`) rather than failing later with a missing
    image or mask.

    Parameters
    ----------
    dataset_dir : str or Path
        Local dataset directory to check.

    Returns
    -------
    list[str]
        Dataset-relative paths that are missing. Empty when the local copy is
        complete. A missing manifest short-circuits the check, since without
        it nothing else can be enumerated.
    """
    root = Path(dataset_dir)

    missing_manifests = [name for name in _REQUIRED_MANIFESTS if not (root / name).is_file()]
    if missing_manifests:
        return missing_manifests

    dataset = AgriStressDataset(root)
    image_ids = dataset.list_image_ids()

    missing = [
        relative_path
        for relative_path in [*image_ids, *dataset.list_instance_ids()]
        if not (root / relative_path).is_file()
    ]
    return missing + _missing_masks(root, image_ids)


def _clusters_with_min_images(dataset: AgriStressDataset, minimum: int) -> list[str]:
    """Return cluster ids holding at least *minimum* images, largest first."""
    sizes = {cluster: len(dataset.list_image_ids(cluster=cluster)) for cluster in dataset.list_clusters()}
    return [cluster for cluster, size in sorted(sizes.items(), key=lambda item: -item[1]) if size >= minimum]


def _clusters_with_instances(dataset: AgriStressDataset) -> list[str]:
    """Return cluster ids whose image metadata parses and that own instance crops, largest first.

    27 of AgriStress-500's 121 clusters have no instance crops at all and a
    further 14 cannot be parsed by ``agristress-utils`` (see
    :func:`sample_cluster_instances`), so a third of the dataset is unusable
    for the Plant->Image and Plant->Plant scenarios. Ordering by crop count
    keeps the clusters these ids are suggested as replacements for from being
    swapped for an equally degenerate one-image cluster.
    """
    sizes: dict[str, int] = {}
    for cluster in dataset.list_clusters():
        try:
            records = dataset.image_metadata(dataset.list_image_ids(cluster=cluster))
        except ValidationError:
            continue
        total_instances = sum(len(record.instances) for record in records)
        if total_instances:
            sizes[cluster] = total_instances
    return [cluster for cluster, _ in sorted(sizes.items(), key=lambda item: -item[1])]


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
        chosen when omitted. Most AgriStress-500 clusters are small (104 of
        121 hold fewer than 8 images, 38 hold exactly one), so a random
        cluster usually yields too few images to evaluate meaningfully.
    num_samples : int
        Number of images to sample, capped to the cluster's size. A warning
        is emitted when the cap bites.
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
    if num_samples > len(image_ids):
        suggestions = _clusters_with_min_images(dataset, num_samples)[:_SUGGESTION_COUNT]
        warnings.warn(
            f"Cluster {chosen!r} holds only {len(image_ids)} image(s), so {len(image_ids)} were sampled "
            f"instead of the requested {num_samples}. Retrieval metrics over so few items saturate at "
            f"1.0000 and measure nothing. Clusters with at least {num_samples} images: {suggestions}.",
            stacklevel=2,
        )

    sample = rng.sample(image_ids, k=min(num_samples, len(image_ids)))
    return chosen, sample


def sample_cluster_with_distractors(
    dataset: AgriStressDataset,
    cluster: str | None = None,
    num_samples: int = 8,
    num_distractor_clusters: int = 6,
    num_distractor_samples: int = 2,
    seed: int = 0,
) -> tuple[str, list[str], list[str]]:
    """Sample one query cluster plus distractor images drawn from other clusters.

    Image->Image retrieval treats same-cluster images as positives, so a
    sample drawn from a single cluster contains no negatives at all and every
    precision and nDCG figure is 1.0000 by construction, however good or bad
    the embeddings are. Mixing in images from other clusters gives the
    ranking something it can get wrong.

    Parameters
    ----------
    dataset : AgriStressDataset
        Loaded dataset to sample from.
    cluster : str or None
        L2 cluster id to use as the query cluster (e.g. ``"A1"``). A random
        cluster is chosen when omitted.
    num_samples : int
        Number of query-cluster images to sample, capped to the cluster's size.
    num_distractor_clusters : int
        Number of other clusters to draw distractors from, capped to the
        number of clusters large enough to supply them.
    num_distractor_samples : int
        Number of images to draw from each distractor cluster.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[str, list[str], list[str]]
        The query cluster id, the query-cluster image ids, and the distractor
        image ids drawn from other clusters.

    Raises
    ------
    ValueError
        If ``cluster`` does not exist in the dataset.
    """
    rng = random.Random(seed)
    chosen, image_ids = sample_cluster(dataset, cluster=cluster, num_samples=num_samples, seed=seed)

    candidates = [
        candidate
        for candidate in dataset.list_clusters()
        if candidate != chosen and len(dataset.list_image_ids(cluster=candidate)) >= num_distractor_samples
    ]
    picked = rng.sample(candidates, k=min(num_distractor_clusters, len(candidates)))

    distractor_ids = [
        image_id
        for candidate in picked
        for image_id in rng.sample(dataset.list_image_ids(cluster=candidate), k=num_distractor_samples)
    ]
    return chosen, image_ids, distractor_ids


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
    parent-plus-instance embedding sets required by the Plant->Image and
    Plant->Plant evaluation scenarios.

    Only 80 of AgriStress-500's 121 clusters are usable here: 27 own no
    instance crops, and 14 cannot be parsed at all. Both are rejected with a
    message naming clusters that do work. When *cluster* is omitted the
    random choice is made among the usable ones only.

    Parameters
    ----------
    dataset : AgriStressDataset
        Loaded dataset to sample from.
    cluster : str or None
        L2 cluster id to sample from (e.g. ``"A1"``). A random usable cluster
        is chosen when omitted. Plant class diversity varies by cluster: some
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
        If ``cluster`` does not exist in the dataset, cannot be parsed, or
        yields no instance crops at all.
    """
    rng = random.Random(seed)

    resolved = cluster
    if resolved is None:
        viable = _clusters_with_instances(dataset)
        if not viable:
            raise ValueError("No cluster in this dataset has usable instance crops.")
        resolved = random.Random(seed).choice(viable)

    chosen, image_ids = sample_cluster(dataset, cluster=resolved, num_samples=num_samples, seed=seed)

    try:
        records = dataset.image_metadata(image_ids)
    except ValidationError as err:
        raise ValueError(
            f"Cluster {chosen!r} has image metadata that agristress-utils cannot parse: AgriStress-500 "
            f"ships images whose `camera_source` is null, but `ImageAttributes.camera_source` is typed as "
            f"a required `str`. 14 of 121 clusters are affected. Use one of these instead: "
            f"{_clusters_with_instances(dataset)[:_SUGGESTION_COUNT]}."
        ) from err

    instance_ids: list[str] = []
    for record in records:
        available = record.instances
        if max_instances_per_image is not None and len(available) > max_instances_per_image:
            available = rng.sample(available, k=max_instances_per_image)
        instance_ids.extend(available)

    if not instance_ids:
        raise ValueError(
            f"Cluster {chosen!r} has no instance crops, so a Plant->Image or Plant->Plant evaluation would "
            f"have no positives to retrieve and would report empty or undefined metrics. 27 of 121 clusters "
            f"are like this. Use one of these instead: {_clusters_with_instances(dataset)[:_SUGGESTION_COUNT]}."
        )

    return chosen, image_ids, instance_ids
