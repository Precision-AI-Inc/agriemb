# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Environment-variable configuration for the API server.

All runtime settings are read from environment variables so the application can
be configured without code changes across environments. Add new settings here
as named functions — never read ``os.environ`` directly in route handlers or
services.
"""

import os


def get_app_env() -> str:
    """Return the current application environment.

    Reads ``PAI_APP_ENV``; defaults to ``"development"``.

    Returns
    -------
    str
        The environment name — typically one of ``"development"``,
        ``"staging"``, or ``"production"``.
    """
    return os.environ.get("PAI_APP_ENV", "development")
