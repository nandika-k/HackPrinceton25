"""
Agent interface module for Care Agent integration.

This module provides a simple, high-level interface for the Care Agent
to query first-aid guidelines without needing to understand the underlying
retrieval system.
"""

from .care_agent import retrieve_guidelines

__all__ = ["retrieve_guidelines"]

