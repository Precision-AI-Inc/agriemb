# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.myproject.metrics.compute."""

from precisionai.myproject.metrics.compute import char_count, word_count


def test_word_count_single_word() -> None:
    assert word_count("hello") == 1


def test_word_count_multiple_words() -> None:
    assert word_count("hello world") == 2


def test_word_count_empty_string() -> None:
    assert word_count("") == 0


def test_word_count_whitespace_only() -> None:
    assert word_count("   ") == 0


def test_char_count_with_spaces() -> None:
    assert char_count("hello world") == 11


def test_char_count_without_spaces() -> None:
    assert char_count("hello world", include_spaces=False) == 10


def test_char_count_empty_string() -> None:
    assert char_count("") == 0
