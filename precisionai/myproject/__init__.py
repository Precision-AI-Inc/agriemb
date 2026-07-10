# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""PAI MyProject — <replace with a one-sentence description of what this project does>.

Package structure
-----------------
- ``precisionai.myproject.metrics``  — pure computation (no I/O, no side effects)
- ``precisionai.myproject.services`` — business logic (orchestrates metrics, handles I/O)
- ``precisionai.myproject.schemas``  — Pydantic v2 request/response models
- ``precisionai.myproject.api``      — FastAPI application factory and route handlers
"""

from precisionai.myproject.services.hello import describe, say_hello

__all__ = [
    "describe",
    "say_hello",
]
