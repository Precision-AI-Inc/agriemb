# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Request and response schemas for the hello endpoints."""

from pydantic import BaseModel, field_validator


class HelloRequest(BaseModel):
    """Payload for the ``POST /v1/hello`` endpoint.

    Attributes
    ----------
    name : str
        The name to greet. Must be non-empty after stripping whitespace.
    shout : bool
        When ``True``, the greeting is returned in uppercase.
    """

    name: str
    shout: bool = False

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only names before they reach the service layer."""
        if not value.strip():
            raise ValueError("name must be non-empty")
        return value


class HelloResponse(BaseModel):
    """Response from the ``POST /v1/hello`` endpoint.

    Attributes
    ----------
    greeting : str
        The computed greeting message.
    word_count : int
        Number of whitespace-delimited tokens in the greeting.
    char_count : int
        Total character count of the greeting (spaces included).
    """

    greeting: str
    word_count: int
    char_count: int
