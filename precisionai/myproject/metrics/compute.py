# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Pure computation utilities for text analysis.

These functions are stateless and take only plain Python values so they can be
called from services, tests, or scripts without any I/O side effects.
"""


def word_count(text: str) -> int:
    """Count the number of whitespace-delimited tokens in a string.

    Parameters
    ----------
    text : str
        The input text to analyse.

    Returns
    -------
    int
        Number of whitespace-delimited tokens. Returns ``0`` for an empty or
        whitespace-only string.

    Examples
    --------
    >>> word_count("hello world")
    2
    >>> word_count("")
    0
    """
    return len(text.split())


def char_count(text: str, *, include_spaces: bool = True) -> int:
    """Count characters in a text string.

    Parameters
    ----------
    text : str
        The input text to analyse.
    include_spaces : bool, optional
        When ``False``, space characters are excluded before counting.

    Returns
    -------
    int
        Character count.

    Examples
    --------
    >>> char_count("hello world")
    11
    >>> char_count("hello world", include_spaces=False)
    10
    """
    return len(text) if include_spaces else len(text.replace(" ", ""))
