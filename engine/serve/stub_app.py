"""Uvicorn import target for the CPU-only echo server."""

from .api import create_stub_app

app = create_stub_app()

__all__ = ["app"]
