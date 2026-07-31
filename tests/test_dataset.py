# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster sampling and dataset verification.

Every test builds a synthetic AgriStress-500 layout on disk, so nothing here
touches the 12 GB real dataset or the hosted embedding API.
"""

import json
import warnings
from pathlib import Path

import pytest

from precisionai.agriemb.dataset import (
    sample_cluster,
    sample_cluster_instances,
    sample_cluster_with_distractors,
    verify_dataset,
)
from precisionai.agristress.utils import AgriStressDataset

# Cluster id -> (image count, instance crops per image). "B1" mirrors the real
# dataset's null `camera_source`, "Z1" its clusters that own no instance crops.
_LAYOUT = {"A1": (12, 3), "C1": (6, 2), "D1": (4, 1), "E1": (3, 1), "B1": (3, 2), "Z1": (5, 0)}


def _image_id(cluster: str, index: int) -> str:
    return f"images/{cluster}/pai-{cluster}{index:03d}.png"


def _instance_id(cluster: str, index: int, ordinal: int) -> str:
    return f"instances/{cluster}/pai-{cluster}{index:03d}-{ordinal}.png"


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Build a synthetic AgriStress-500 dataset directory and return its root."""
    i2i: dict[str, dict] = {}
    instance_to_image: dict[str, list[str]] = {}
    instance_labels: dict[str, str] = {}

    for cluster, (n_images, n_instances) in _LAYOUT.items():
        images = [_image_id(cluster, i) for i in range(n_images)]
        i2i[cluster] = {
            "images": images,
            "class_name": cluster[0],
            # Only B1 reproduces the null `camera_source` that agristress-utils
            # cannot parse (14 of 121 real clusters are affected).
            "attributes": {
                "plants": ["Crop | Wheat"],
                "time_period": "noon",
                "camera_model": "test-cam",
                "camera_source": None if cluster == "B1" else "drone",
                "collecting_date": "2026-01-01",
            },
        }
        for i, image in enumerate(images):
            instances = [_instance_id(cluster, i, ordinal) for ordinal in range(n_instances)]
            if instances:
                instance_to_image[image] = instances
            for instance in instances:
                instance_labels[instance] = "Crop | Wheat"

    (tmp_path / "image2image.json").write_text(json.dumps({"metadata": i2i}))
    (tmp_path / "plant2image.json").write_text(json.dumps({"instance_to_image": instance_to_image}))
    (tmp_path / "plant2plant.json").write_text(json.dumps({"instance_labels": instance_labels}))
    (tmp_path / "class_map.json").write_text(json.dumps({"classes": []}))

    for relative_path in [*instance_labels, *(img for group in i2i.values() for img in group["images"])]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        if relative_path.startswith("images/"):
            mask = tmp_path / "masks" / relative_path.split("/", 1)[1]
            mask.parent.mkdir(parents=True, exist_ok=True)
            mask.write_bytes(b"")

    return tmp_path


@pytest.fixture
def dataset(dataset_root: Path) -> AgriStressDataset:
    """Load the synthetic dataset."""
    return AgriStressDataset(dataset_root)


class TestSampleCluster:
    """Tests for :func:`precisionai.agriemb.dataset.sample_cluster`."""

    def test_samples_requested_count_from_named_cluster(self, dataset: AgriStressDataset) -> None:
        cluster, sample = sample_cluster(dataset, cluster="A1", num_samples=5)

        assert cluster == "A1"
        assert len(sample) == 5
        assert len(set(sample)) == 5
        assert set(sample) <= set(dataset.list_image_ids(cluster="A1"))

    def test_is_reproducible_for_a_given_seed(self, dataset: AgriStressDataset) -> None:
        first = sample_cluster(dataset, cluster="A1", num_samples=5, seed=7)
        second = sample_cluster(dataset, cluster="A1", num_samples=5, seed=7)
        other = sample_cluster(dataset, cluster="A1", num_samples=5, seed=8)

        assert first == second
        assert first != other

    def test_rejects_an_unknown_cluster(self, dataset: AgriStressDataset) -> None:
        with pytest.raises(ValueError, match="Unknown cluster"):
            sample_cluster(dataset, cluster="NOPE")

    def test_warns_and_caps_when_the_cluster_is_too_small(self, dataset: AgriStressDataset) -> None:
        with pytest.warns(UserWarning, match="holds only 3 image"):
            _, sample = sample_cluster(dataset, cluster="E1", num_samples=8)

        assert len(sample) == 3

    def test_does_not_warn_when_the_cluster_is_large_enough(self, dataset: AgriStressDataset) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample_cluster(dataset, cluster="A1", num_samples=8)


class TestSampleClusterWithDistractors:
    """Tests for :func:`precisionai.agriemb.dataset.sample_cluster_with_distractors`."""

    def test_draws_distractors_from_other_clusters(self, dataset: AgriStressDataset) -> None:
        cluster, sample, distractors = sample_cluster_with_distractors(
            dataset, cluster="A1", num_samples=6, num_distractor_clusters=3, num_distractor_samples=2
        )

        assert cluster == "A1"
        assert len(sample) == 6
        assert len(distractors) == 6
        assert not set(sample) & set(distractors)

        distractor_clusters = {distractor.split("/")[1] for distractor in distractors}
        assert len(distractor_clusters) == 3
        assert "A1" not in distractor_clusters

    def test_caps_distractor_clusters_to_what_the_dataset_offers(self, dataset: AgriStressDataset) -> None:
        _, _, distractors = sample_cluster_with_distractors(
            dataset, cluster="A1", num_samples=2, num_distractor_clusters=99, num_distractor_samples=2
        )

        # Every cluster but A1 has at least two images, so all five can supply distractors.
        assert {distractor.split("/")[1] for distractor in distractors} == set(_LAYOUT) - {"A1"}


class TestSampleClusterInstances:
    """Tests for :func:`precisionai.agriemb.dataset.sample_cluster_instances`."""

    def test_collects_instance_crops_for_the_sampled_parents(self, dataset: AgriStressDataset) -> None:
        cluster, images, instances = sample_cluster_instances(
            dataset, cluster="A1", num_samples=3, max_instances_per_image=2
        )

        assert cluster == "A1"
        assert len(images) == 3
        assert len(instances) == 6
        assert all(instance.startswith("instances/A1/") for instance in instances)

    def test_keeps_every_crop_when_uncapped(self, dataset: AgriStressDataset) -> None:
        _, _, instances = sample_cluster_instances(dataset, cluster="C1", num_samples=2, max_instances_per_image=None)

        assert len(instances) == 4

    def test_rejects_a_cluster_with_no_instance_crops(self, dataset: AgriStressDataset) -> None:
        with pytest.raises(ValueError, match="no instance crops"):
            sample_cluster_instances(dataset, cluster="Z1", num_samples=2)

    def test_names_viable_clusters_when_rejecting(self, dataset: AgriStressDataset) -> None:
        with pytest.raises(ValueError, match="Use one of these instead") as excinfo:
            sample_cluster_instances(dataset, cluster="Z1", num_samples=2)

        assert "A1" in str(excinfo.value)
        assert "Z1" not in str(excinfo.value).split("Use one of these instead:")[1]

    def test_reports_unparseable_metadata_instead_of_a_pydantic_traceback(self, dataset: AgriStressDataset) -> None:
        with pytest.raises(ValueError, match="camera_source"):
            sample_cluster_instances(dataset, cluster="B1", num_samples=2)

    def test_random_choice_only_picks_usable_clusters(self, dataset: AgriStressDataset) -> None:
        chosen = {sample_cluster_instances(dataset, num_samples=2, seed=seed)[0] for seed in range(25)}

        assert chosen <= {"A1", "C1", "D1", "E1"}
        assert not chosen & {"B1", "Z1"}


class TestVerifyDataset:
    """Tests for :func:`precisionai.agriemb.dataset.verify_dataset`."""

    def test_reports_nothing_missing_for_a_complete_dataset(self, dataset_root: Path) -> None:
        assert verify_dataset(dataset_root) == []

    def test_reports_a_missing_image(self, dataset_root: Path) -> None:
        (dataset_root / _image_id("A1", 0)).unlink()

        assert verify_dataset(dataset_root) == [_image_id("A1", 0)]

    def test_reports_a_missing_mask(self, dataset_root: Path) -> None:
        (dataset_root / "masks" / "A1" / "pai-A1000.png").unlink()

        assert verify_dataset(dataset_root) == ["masks/A1/pai-A1000.png"]

    def test_reports_a_missing_instance_crop(self, dataset_root: Path) -> None:
        (dataset_root / _instance_id("A1", 0, 0)).unlink()

        assert verify_dataset(dataset_root) == [_instance_id("A1", 0, 0)]

    def test_short_circuits_on_a_missing_manifest(self, dataset_root: Path) -> None:
        (dataset_root / "plant2image.json").unlink()

        assert verify_dataset(dataset_root) == ["plant2image.json"]

    def test_reports_every_manifest_for_an_empty_directory(self, tmp_path: Path) -> None:
        assert verify_dataset(tmp_path) == [
            "image2image.json",
            "plant2image.json",
            "plant2plant.json",
            "class_map.json",
        ]
