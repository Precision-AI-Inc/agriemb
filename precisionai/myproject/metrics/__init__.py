# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Precision AI MyProject metrics — public re-exports.

All stable public symbols are re-exported here so callers can import directly
from ``precisionai.myproject.metrics`` without knowing the submodule layout.
"""

from precisionai.myproject.metrics.compute import char_count, word_count

__all__ = [
    "char_count",
    "word_count",
]
