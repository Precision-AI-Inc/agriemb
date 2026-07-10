# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.myproject.services.hello."""

import pytest

from precisionai.myproject.services.hello import describe, say_hello


def test_say_hello_returns_greeting(sample_name: str) -> None:
    assert say_hello(sample_name) == "Hello, World!"


def test_say_hello_shout(sample_name: str) -> None:
    assert say_hello(sample_name, shout=True) == "HELLO, WORLD!"


def test_say_hello_preserves_name_casing() -> None:
    assert say_hello("Precision AI") == "Hello, Precision AI!"


def test_say_hello_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        say_hello("")


def test_say_hello_whitespace_only_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        say_hello("   ")


def test_describe_returns_correct_keys(sample_name: str) -> None:
    result = describe(sample_name)
    assert set(result.keys()) == {"greeting", "word_count", "char_count"}


def test_describe_greeting_value(sample_name: str) -> None:
    result = describe(sample_name)
    assert result["greeting"] == "Hello, World!"


def test_describe_word_count(sample_name: str) -> None:
    result = describe(sample_name)
    assert result["word_count"] == 2


def test_describe_char_count(sample_name: str) -> None:
    result = describe(sample_name)
    assert result["char_count"] == len("Hello, World!")


def test_describe_shout(sample_name: str) -> None:
    result = describe(sample_name, shout=True)
    assert result["greeting"] == "HELLO, WORLD!"
