"""
API Module

This module contains the FastAPI application for serving RAG retrieval requests.
"""

from .main import app, router

__all__ = ['app', 'router']

