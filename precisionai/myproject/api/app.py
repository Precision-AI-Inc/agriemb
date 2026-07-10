# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory and CLI entry point.

Usage
-----
Start the development server::

    pai - myproject

Or run the module directly::

    python -m precisionai.myproject.api.app --host 0.0.0.0 --port 8000

Interactive API docs are available at ``http://localhost:8000/docs`` once the
server is running.
"""

import argparse

import uvicorn
from fastapi import FastAPI

from precisionai.myproject.api.routes.hello import router


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Returns
    -------
    FastAPI
        The configured application instance with all routers registered.
    """
    app = FastAPI(
        title="PAI MyProject",
        version="0.1.0",
        description="Precision AI project template API.",
    )
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Launch the uvicorn server from the CLI entry point."""
    parser = argparse.ArgumentParser(description="PAI MyProject API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable live reload")
    args = parser.parse_args()

    uvicorn.run(
        "precisionai.myproject.api.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
