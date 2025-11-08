"""
Main entry point for testing the first-aid guidelines retrieval system.

This is an optional script for testing and demonstration purposes.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent_interface import retrieve_guidelines
from agent_interface.care_agent import get_available_scenarios


def main():
    """
    Demonstrate the retrieval system functionality.
    """
    print("=" * 60)
    print("First-Aid Guidelines Retrieval System - Demo")
    print("=" * 60)
    print()
    
    # List available scenarios
    print("Available scenarios:")
    scenarios = get_available_scenarios()
    for scenario in scenarios:
        print(f"  - {scenario}")
    print()
    
    # Retrieve guidelines for cardiac arrest (adult)
    print("Retrieving guidelines for: cardiac_arrest (adult)")
    print("-" * 60)
    guidelines = retrieve_guidelines(scenario="cardiac_arrest", age_group="adult")
    
    print(f"Text preview (first 200 chars):")
    print(guidelines["text"][:200] + "...")
    print()
    
    print(f"Images: {guidelines['images']}")
    print()
    
    print(f"Metadata: {guidelines['metadata']}")
    print()
    
    # Retrieve guidelines for choking (child)
    print("Retrieving guidelines for: choking (child)")
    print("-" * 60)
    guidelines = retrieve_guidelines(scenario="choking", age_group="child")
    
    print(f"Text preview (first 200 chars):")
    print(guidelines["text"][:200] + "...")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

