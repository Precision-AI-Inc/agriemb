# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the FastAPI application."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from precisionai.myproject.api.app import main
from precisionai.myproject.api.config import get_app_env


def test_hello_endpoint_success(client: TestClient) -> None:
    response = client.post("/v1/hello", json={"name": "World"})
    assert response.status_code == 200
    data = response.json()
    assert data["greeting"] == "Hello, World!"
    assert data["word_count"] == 2
    assert data["char_count"] == len("Hello, World!")


def test_hello_endpoint_shout(client: TestClient) -> None:
    response = client.post("/v1/hello", json={"name": "World", "shout": True})
    assert response.status_code == 200
    assert response.json()["greeting"] == "HELLO, WORLD!"


def test_hello_endpoint_blank_name_rejected(client: TestClient) -> None:
    response = client.post("/v1/hello", json={"name": "   "})
    assert response.status_code == 422


def test_hello_endpoint_missing_name_rejected(client: TestClient) -> None:
    response = client.post("/v1/hello", json={})
    assert response.status_code == 422


def test_get_app_env_default() -> None:
    os.environ.pop("PAI_APP_ENV", None)
    assert get_app_env() == "development"


def test_get_app_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAI_APP_ENV", "production")
    assert get_app_env() == "production"


def test_main_default_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pai-myproject"])
    with patch("uvicorn.run") as mock_run:
        main()
    mock_run.assert_called_once_with(
        "precisionai.myproject.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def test_main_no_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pai-myproject", "--no-reload"])
    with patch("uvicorn.run") as mock_run:
        main()
    mock_run.assert_called_once_with(
        "precisionai.myproject.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


def test_main_custom_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pai-myproject", "--host", "127.0.0.1", "--port", "9000"])
    with patch("uvicorn.run") as mock_run:
        main()
    mock_run.assert_called_once_with(
        "precisionai.myproject.api.app:app",
        host="127.0.0.1",
        port=9000,
        reload=True,
    )
