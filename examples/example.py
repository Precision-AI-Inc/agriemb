# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Standalone example: call the hello service and metrics without the API server.

This script demonstrates calling the service layer directly from Python,
bypassing the HTTP API. Use it as a reference for scripting, notebooks, or
one-off evaluations.

Run from the project root::

    python examples/example.py
    python examples/example.py --name "Precision AI" --shout
"""

import argparse

from precisionai.myproject.metrics.compute import char_count, word_count
from precisionai.myproject.services.hello import say_hello


def main() -> None:
    """Run the standalone hello example."""
    parser = argparse.ArgumentParser(description="PAI MyProject — hello example")
    parser.add_argument("--name", default="World", help="Name to greet (default: World)")
    parser.add_argument("--shout", action="store_true", help="Return the greeting in uppercase")
    args = parser.parse_args()

    greeting = say_hello(args.name, shout=args.shout)

    print(f"Greeting          : {greeting}")
    print(f"Words             : {word_count(greeting)}")
    print(f"Chars (w/ spaces) : {char_count(greeting)}")
    print(f"Chars (no spaces) : {char_count(greeting, include_spaces=False)}")


if __name__ == "__main__":
    main()
