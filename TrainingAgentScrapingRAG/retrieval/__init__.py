"""
Retrieval module for scenario-based first-aid guidance retrieval.

This module provides functions to load and retrieve curated first-aid guidelines
from the knowledge base, organized by scenario and age group.
"""

from .knowledge_loader import (
    load_knowledge_base,
    get_guidelines,
    list_scenarios,
    add_new_scenario,
)

__all__ = [
    "load_knowledge_base",
    "get_guidelines",
    "list_scenarios",
    "add_new_scenario",
]

