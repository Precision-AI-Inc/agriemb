# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for data-URI encoding and batched embedding.

``embed_images`` is exercised against a test double rather than the hosted
API, so nothing here makes a network call.
"""

import base64
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from precisionai.agriemb.embeddings import embed_images, image_array_to_data_uri, image_to_data_uri


class _StubEmbeddings:
    """Records every ``create`` call and returns one vector per input."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str]) -> Any:
        self.calls.append(list(input))
        data = [type("Item", (), {"embedding": [float(len(self.calls)), float(i)]})() for i in range(len(input))]
        return type("Response", (), {"data": data})()


class _StubClient:
    """Minimal stand-in for ``openai.OpenAI`` satisfying ``EmbeddingsClient``."""

    def __init__(self) -> None:
        self.embeddings = _StubEmbeddings()


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    """Write a tiny PNG and return its path."""
    path = tmp_path / "tile.png"
    Image.new("RGB", (4, 4), "green").save(path)
    return path


class TestImageToDataUri:
    """Tests for :func:`precisionai.agriemb.embeddings.image_to_data_uri`."""

    def test_encodes_a_png_as_a_base64_data_uri(self, png_path: Path) -> None:
        uri = image_to_data_uri(png_path)

        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == png_path.read_bytes()

    def test_uses_the_mime_type_implied_by_the_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "field.jpg"
        Image.new("RGB", (4, 4), "blue").save(path)

        assert image_to_data_uri(path).startswith("data:image/jpeg;base64,")

    def test_rejects_a_file_with_no_image_mime_type(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.unknownext"
        path.write_bytes(b"x")

        with pytest.raises(ValueError, match="image MIME type"):
            image_to_data_uri(path)


class TestImageArrayToDataUri:
    """Tests for :func:`precisionai.agriemb.embeddings.image_array_to_data_uri`."""

    def test_encodes_an_in_memory_image(self) -> None:
        uri = image_array_to_data_uri(Image.new("RGB", (4, 4), "red"))

        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"

    def test_honours_the_requested_format(self) -> None:
        uri = image_array_to_data_uri(Image.new("RGB", (4, 4), "red"), image_format="JPEG")

        assert uri.startswith("data:image/jpeg;base64,")


class TestEmbedImages:
    """Tests for :func:`precisionai.agriemb.embeddings.embed_images`."""

    def test_returns_one_vector_per_image_id(self, png_path: Path) -> None:
        client = _StubClient()

        embeddings = embed_images(client, {"a": png_path, "b": png_path}, progress=False)

        assert sorted(embeddings) == ["a", "b"]
        assert all(len(vector) == 2 for vector in embeddings.values())

    def test_splits_the_work_into_batches(self, png_path: Path) -> None:
        client = _StubClient()
        images = {f"id-{i}": png_path for i in range(5)}

        embed_images(client, images, batch_size=2, progress=False)

        assert [len(call) for call in client.embeddings.calls] == [2, 2, 1]

    def test_accepts_in_memory_tiles_alongside_paths(self, png_path: Path) -> None:
        client = _StubClient()

        embeddings = embed_images(
            client, {"file": png_path, "tile": Image.new("RGB", (4, 4), "red")}, batch_size=1, progress=False
        )

        assert sorted(embeddings) == ["file", "tile"]
        assert all(uri[0].startswith("data:image/") for uri in client.embeddings.calls)

    def test_returns_nothing_for_no_input(self) -> None:
        client = _StubClient()

        assert embed_images(client, {}, progress=False) == {}
        assert client.embeddings.calls == []
