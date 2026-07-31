# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dense Patch Token tiling."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from precisionai.agriemb.dpt import DEFAULT_GRID_SIZE, embed_patch_tokens, tile_image


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    """Write a 64x32 RGB image and return its path."""
    path = tmp_path / "field.png"
    Image.new("RGB", (64, 32), "green").save(path)
    return path


class TestTileImage:
    """Tests for :func:`precisionai.agriemb.dpt.tile_image`."""

    def test_produces_one_tile_per_grid_cell(self, image_path: Path) -> None:
        tiles = tile_image(image_path, (4, 8))

        assert len(tiles) == 32
        assert set(tiles) == {f"{row},{col}" for row in range(4) for col in range(8)}

    def test_tiles_partition_the_image_exactly(self, image_path: Path) -> None:
        tiles = tile_image(image_path, (4, 8))

        # 64 wide / 8 cols = 8px, 32 high / 4 rows = 8px.
        assert all(tile.size == (8, 8) for tile in tiles.values())
        assert sum(tile.size[0] * tile.size[1] for tile in tiles.values()) == 64 * 32

    def test_covers_every_pixel_when_the_grid_does_not_divide_evenly(self, tmp_path: Path) -> None:
        path = tmp_path / "odd.png"
        Image.new("RGB", (10, 7), "blue").save(path)

        tiles = tile_image(path, (3, 3))

        assert len(tiles) == 9
        assert sum(tile.size[0] * tile.size[1] for tile in tiles.values()) == 10 * 7

    def test_a_one_by_one_grid_returns_the_whole_image(self, image_path: Path) -> None:
        tiles = tile_image(image_path, (1, 1))

        assert list(tiles) == ["0,0"]
        assert tiles["0,0"].size == (64, 32)

    @pytest.mark.parametrize("grid_size", [(0, 8), (8, 0), (0, 0), (-1, 8)])
    def test_rejects_a_grid_dimension_below_one(self, image_path: Path, grid_size: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="at least 1 in both dimensions"):
            tile_image(image_path, grid_size)

    def test_default_grid_is_fine_enough_for_label_aware_metrics(self) -> None:
        # A coarser grid collapses every AgriStress-500 patch label to
        # `background` on most clusters, disabling every separation metric.
        assert DEFAULT_GRID_SIZE == (8, 8)


class _UnusedClient:
    """Client that fails the test if the embedding API is reached at all."""

    @property
    def embeddings(self) -> Any:
        raise AssertionError("no embedding call should be made for a degenerate input")


class TestEmbedPatchTokens:
    """Tests for :func:`precisionai.agriemb.dpt.embed_patch_tokens`."""

    def test_rejects_an_empty_input_before_calling_the_api(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            embed_patch_tokens(_UnusedClient(), {})

    def test_rejects_a_zero_grid_dimension_before_calling_the_api(self, image_path: Path) -> None:
        # `--grid-height 0` would otherwise embed nothing and then fail with a
        # bare StopIteration no example script's `except` clauses catch.
        with pytest.raises(ValueError, match="at least 1 in both dimensions"):
            embed_patch_tokens(_UnusedClient(), {"field": image_path}, grid_size=(0, 8))
