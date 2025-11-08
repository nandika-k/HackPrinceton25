"""
API module for serving first-aid guidelines via FastAPI.

This module provides REST API endpoints for accessing the first-aid
guidelines retrieval system. Useful for integration with web applications
or external services.
"""

from .endpoints import app, router

__all__ = ["app", "router"]

