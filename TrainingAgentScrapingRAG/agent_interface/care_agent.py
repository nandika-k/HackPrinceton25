"""
Care Agent interface for retrieving first-aid guidelines.

This module provides a simplified interface that the Care Agent can use
to query first-aid guidelines. It abstracts away the details of the
retrieval system and provides a clean API.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from ..retrieval.knowledge_loader import get_guidelines, list_scenarios


def retrieve_guidelines(
    scenario: str,
    age_group: str = "adult",
    knowledge_base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Retrieve first-aid guidelines for a given scenario and age group.
    
    This is the main function that the Care Agent should call to get
    first-aid guidance. It returns a structured dictionary with text,
    images, and metadata.
    
    Args:
        scenario: Name of the emergency scenario.
                  Examples: "cardiac_arrest", "choking", "stroke"
        age_group: Age group for the guidelines ("adult" or "child").
                   Defaults to "adult".
        knowledge_base_path: Optional path to knowledge base directory.
                            Used primarily for testing or custom setups.
    
    Returns:
        Dictionary with the following structure:
        {
            "text": str,              # Text content of guidelines
            "images": List[str],      # List of image file paths
            "metadata": Dict          # Metadata (scenario, age_group, source, etc.)
        }
        
        If scenario is not found, returns empty structure with error in metadata.
    
    Example:
        >>> guidelines = retrieve_guidelines(
        ...     scenario="cardiac_arrest",
        ...     age_group="adult"
        ... )
        >>> print(guidelines["text"])
        # Cardiac Arrest - Adult First Aid Guidelines
        ...
        >>> print(guidelines["images"])
        ['cardiac_arrest/adult/images/cpr_diagram.png']
        >>> print(guidelines["metadata"])
        {'scenario': 'cardiac_arrest', 'age_group': 'adult', ...}
    
    TODO: Add input validation and sanitization.
    TODO: Add logging for agent queries.
    TODO: Add support for scenario aliases or fuzzy matching.
    TODO: Add caching for frequently accessed scenarios.
    TODO: Add rate limiting if needed for API usage.
    """
    # Delegate to the knowledge_loader module
    return get_guidelines(
        scenario=scenario,
        age_group=age_group,
        base_path=knowledge_base_path
    )


def get_available_scenarios(knowledge_base_path: Optional[Path] = None) -> list:
    """
    Get a list of all available emergency scenarios.
    
    Helper function for the Care Agent to discover what scenarios
    are available in the knowledge base.
    
    Args:
        knowledge_base_path: Optional path to knowledge base directory.
    
    Returns:
        List of scenario names (strings).
    
    Example:
        >>> scenarios = get_available_scenarios()
        >>> print(scenarios)
        ['cardiac_arrest', 'choking']
    
    TODO: Add filtering or search capabilities.
    """
    return list_scenarios(base_path=knowledge_base_path)

