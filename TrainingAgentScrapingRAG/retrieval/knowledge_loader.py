"""
Knowledge loader module for retrieving first-aid guidelines.

This module provides scenario-based retrieval of curated first-aid documents
without using embeddings or semantic search. All retrieval is based on
direct file access organized by scenario and age group.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


# Base path to knowledge base directory
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base"


def load_knowledge_base(base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load all knowledge base content from the knowledge_base directory.
    
    Scans the knowledge_base directory and returns a structured dictionary
    containing all available scenarios, age groups, and their associated
    text and image files.
    
    Args:
        base_path: Optional path to knowledge base directory.
                   Defaults to knowledge_base/ relative to this module.
    
    Returns:
        Dictionary with structure:
        {
            "scenario_name": {
                "adult": {
                    "text": str,
                    "images": List[str],
                    "metadata": Dict
                },
                "child": {
                    "text": str,
                    "images": List[str],
                    "metadata": Dict
                }
            },
            ...
        }
    
    TODO: Add error handling for missing files or malformed directories.
    TODO: Add support for metadata files (JSON) if needed.
    """
    if base_path is None:
        base_path = KNOWLEDGE_BASE_PATH
    
    knowledge_base = {}
    
    if not base_path.exists():
        # TODO: Create directory structure if it doesn't exist
        return knowledge_base
    
    # Iterate through scenario directories
    for scenario_dir in base_path.iterdir():
        if not scenario_dir.is_dir() or scenario_dir.name.startswith('.'):
            continue
        
        scenario_name = scenario_dir.name
        knowledge_base[scenario_name] = {}
        
        # Check for adult and child subdirectories
        for age_group in ["adult", "child"]:
            age_group_path = scenario_dir / age_group
            
            if not age_group_path.exists():
                continue
            
            # Load text guidelines
            guidelines_file = age_group_path / "guidelines.txt"
            text_content = ""
            if guidelines_file.exists():
                with open(guidelines_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            # Load images
            images_dir = age_group_path / "images"
            image_files = []
            if images_dir.exists():
                # TODO: Support multiple image formats (PNG, JPG, SVG, etc.)
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
                for img_file in images_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        # Store relative path from knowledge_base
                        image_files.append(str(img_file.relative_to(base_path)))
            
            # Load metadata if available
            metadata_file = age_group_path / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                # Default metadata
                metadata = {
                    "scenario": scenario_name,
                    "age_group": age_group,
                    "last_updated": None,
                    "source": None
                }
            
            knowledge_base[scenario_name][age_group] = {
                "text": text_content,
                "images": image_files,
                "metadata": metadata
            }
    
    return knowledge_base


def get_guidelines(scenario: str, age_group: str = "adult", base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Retrieve guidelines for a specific scenario and age group.
    
    Returns a dictionary containing text instructions, associated images,
    and metadata for the specified scenario and age group.
    
    Args:
        scenario: Name of the emergency scenario (e.g., "cardiac_arrest", "choking").
        age_group: Age group for the guidelines ("adult" or "child"). Defaults to "adult".
        base_path: Optional path to knowledge base directory.
    
    Returns:
        Dictionary with keys:
        - "text": str - Text content of the guidelines
        - "images": List[str] - List of relative paths to associated images
        - "metadata": Dict - Metadata about the guidelines (source, last_updated, etc.)
        
        Returns empty dictionary with empty values if scenario/age_group not found.
    
    Example:
        >>> guidelines = get_guidelines("cardiac_arrest", "adult")
        >>> print(guidelines["text"])
        # Cardiac Arrest - Adult First Aid Guidelines
        ...
    
    TODO: Add validation for scenario and age_group parameters.
    TODO: Add caching mechanism if performance becomes an issue.
    TODO: Add support for partial matches or scenario aliases.
    """
    if base_path is None:
        base_path = KNOWLEDGE_BASE_PATH
    
    # Normalize inputs
    scenario = scenario.lower().strip()
    age_group = age_group.lower().strip()
    
    # Validate age_group
    if age_group not in ["adult", "child"]:
        # TODO: Log warning about invalid age_group
        age_group = "adult"  # Default fallback
    
    # Construct path to guidelines
    scenario_path = base_path / scenario / age_group
    
    if not scenario_path.exists():
        # Return empty structure if not found
        return {
            "text": "",
            "images": [],
            "metadata": {
                "scenario": scenario,
                "age_group": age_group,
                "error": "Scenario or age group not found"
            }
        }
    
    # Load text guidelines
    guidelines_file = scenario_path / "guidelines.txt"
    text_content = ""
    if guidelines_file.exists():
        with open(guidelines_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    
    # Load images
    images_dir = scenario_path / "images"
    image_files = []
    if images_dir.exists():
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                image_files.append(str(img_file.relative_to(base_path)))
    
    # Load metadata
    metadata_file = scenario_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "scenario": scenario,
            "age_group": age_group,
            "last_updated": None,
            "source": None
        }
    
    return {
        "text": text_content,
        "images": image_files,
        "metadata": metadata
    }


def list_scenarios(base_path: Optional[Path] = None) -> List[str]:
    """
    List all available emergency scenarios in the knowledge base.
    
    Scans the knowledge_base directory and returns a list of all
    scenario names that have been curated.
    
    Args:
        base_path: Optional path to knowledge base directory.
    
    Returns:
        List of scenario names (strings), sorted alphabetically.
    
    Example:
        >>> scenarios = list_scenarios()
        >>> print(scenarios)
        ['cardiac_arrest', 'choking']
    
    TODO: Add filtering by age group availability.
    TODO: Add option to return detailed scenario information.
    """
    if base_path is None:
        base_path = KNOWLEDGE_BASE_PATH
    
    scenarios = []
    
    if not base_path.exists():
        return scenarios
    
    # Scan directory for scenario folders
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has at least one age group subdirectory
            has_adult = (item / "adult").exists()
            has_child = (item / "child").exists()
            
            if has_adult or has_child:
                scenarios.append(item.name)
    
    return sorted(scenarios)


def add_new_scenario(
    scenario_name: str,
    text_files: Optional[List[str]] = None,
    image_files: Optional[List[str]] = None,
    base_path: Optional[Path] = None
) -> bool:
    """
    Add a new scenario to the knowledge base.
    
    Creates directory structure and placeholder files for a new emergency
    scenario. This is a placeholder function for adding new official guidance.
    
    Args:
        scenario_name: Name of the new scenario (e.g., "burns", "stroke").
        text_files: Optional list of text file paths to copy (not implemented yet).
        image_files: Optional list of image file paths to copy (not implemented yet).
        base_path: Optional path to knowledge base directory.
    
    Returns:
        True if scenario structure was created successfully, False otherwise.
    
    TODO: Implement file copying from text_files and image_files.
    TODO: Add validation for scenario_name (no special characters, etc.).
    TODO: Add support for metadata.json creation.
    TODO: Add error handling and logging.
    
    Example:
        >>> add_new_scenario("stroke")
        True
        >>> # Creates: knowledge_base/stroke/adult/guidelines.txt
        >>> #         knowledge_base/stroke/child/guidelines.txt
    """
    if base_path is None:
        base_path = KNOWLEDGE_BASE_PATH
    
    # Normalize scenario name
    scenario_name = scenario_name.lower().strip().replace(" ", "_")
    
    # Create base scenario directory
    scenario_path = base_path / scenario_name
    scenario_path.mkdir(parents=True, exist_ok=True)
    
    # Create age group directories and placeholder files
    for age_group in ["adult", "child"]:
        age_group_path = scenario_path / age_group
        age_group_path.mkdir(parents=True, exist_ok=True)
        
        # Create images directory
        images_dir = age_group_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder guidelines file if it doesn't exist
        guidelines_file = age_group_path / "guidelines.txt"
        if not guidelines_file.exists():
            placeholder_text = f"# {scenario_name.replace('_', ' ').title()} - {age_group.title()} First Aid Guidelines\n\n## Overview\n[TODO: Add official guidelines here]\n\n## Steps to Follow\n\n[TODO: Add step-by-step instructions]\n\n## Important Notes\n\n[TODO: Add important safety notes]\n"
            with open(guidelines_file, 'w', encoding='utf-8') as f:
                f.write(placeholder_text)
        
        # TODO: Copy files from text_files if provided
        # TODO: Copy files from image_files if provided
    
    return True

