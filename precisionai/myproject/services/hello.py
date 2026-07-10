# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Business logic for greeting operations.

This module is the single authoritative place for the greeting computation.
Route handlers and examples both delegate here — never inline the logic in the
API layer.
"""

from precisionai.myproject.metrics.compute import char_count, word_count


def say_hello(name: str, *, shout: bool = False) -> str:
    """Compute a greeting message for the given name.

    Parameters
    ----------
    name : str
        The name to include in the greeting. Must be non-empty after stripping
        leading and trailing whitespace.
    shout : bool, optional
        When ``True``, return the greeting in uppercase.

    Returns
    -------
    str
        The greeting message, optionally uppercased.

    Raises
    ------
    ValueError
        If *name* is empty or contains only whitespace.

    Examples
    --------
    >>> say_hello("World")
    'Hello, World!'
    >>> say_hello("World", shout=True)
    'HELLO, WORLD!'
    """
    if not name.strip():
        raise ValueError("name must be non-empty") from None
    greeting = f"Hello, {name}!"
    return greeting.upper() if shout else greeting


def describe(name: str, *, shout: bool = False) -> dict[str, object]:
    """Return a greeting together with basic text statistics.

    Combines :func:`say_hello` with the metrics layer to produce a
    self-contained result dict suitable for serialisation.

    Parameters
    ----------
    name : str
        The name to greet. Forwarded to :func:`say_hello` unchanged.
    shout : bool, optional
        Passed through to :func:`say_hello`.

    Returns
    -------
    dict[str, object]
        Keys: ``greeting`` (str), ``word_count`` (int), ``char_count`` (int).

    Raises
    ------
    ValueError
        If *name* is empty or contains only whitespace.
    """
    greeting = say_hello(name, shout=shout)
    return {
        "greeting": greeting,
        "word_count": word_count(greeting),
        "char_count": char_count(greeting),
    }
