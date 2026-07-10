# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Precision AI MyProject services — public re-exports."""

from precisionai.myproject.services.hello import describe, say_hello

__all__ = [
    "describe",
    "say_hello",
]
