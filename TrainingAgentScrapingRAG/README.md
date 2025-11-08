# First-Aid Guidelines Retrieval System

A Python framework for scenario-based retrieval of curated first-aid guidance documents, designed for healthcare AI agents providing first-aid guidance for cardiac emergencies and other medical situations.

## Overview

This system provides a **scenario-based retrieval system** using only curated documents, with **no embeddings or semantic search**. All retrieval is based on direct file access organized by scenario and age group.

## Project Structure

```
TrainingAgentScrapingRAG/
├── knowledge_base/          # Curated text and images, organized by scenario
│   ├── cardiac_arrest/
│   │   ├── adult/
│   │   │   ├── guidelines.txt
│   │   │   └── images/
│   │   └── child/
│   │       ├── guidelines.txt
│   │       └── images/
│   └── choking/
│       ├── adult/
│       └── child/
├── retrieval/               # Functions to fetch instructions based on scenario
│   ├── __init__.py
│   └── knowledge_loader.py
├── agent_interface/         # Methods for the Care Agent to query guidelines
│   ├── __init__.py
│   └── care_agent.py
├── api/                     # Optional FastAPI scaffold to serve retrieval functions
│   ├── __init__.py
│   └── endpoints.py
├── requirements.txt
└── README.md
```

## Features

- **Scenario-based retrieval**: Direct access to guidelines by scenario name and age group
- **No embeddings or semantic search**: Pure file-based retrieval from curated documents
- **Modular design**: Easy to extend with new scenarios and age groups
- **Agent-friendly API**: Simple function calls for the Care Agent
- **Optional REST API**: FastAPI endpoints for web integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The knowledge base is already populated with sample scenarios (cardiac_arrest, choking).

## Quick Start

### Basic Usage (Python)

```python
from agent_interface import retrieve_guidelines

# Retrieve guidelines for cardiac arrest in adults
guidelines = retrieve_guidelines(scenario="cardiac_arrest", age_group="adult")

print(guidelines["text"])        # Text content
print(guidelines["images"])      # List of image paths
print(guidelines["metadata"])    # Metadata dictionary
```

### List Available Scenarios

```python
from agent_interface.care_agent import get_available_scenarios

scenarios = get_available_scenarios()
print(scenarios)  # ['cardiac_arrest', 'choking']
```

### Using the API (Optional)

1. Start the FastAPI server:
```bash
uvicorn api.endpoints:app --reload
```

2. Access the API:
- API Docs: http://localhost:8000/docs
- Get guidelines: `GET /api/v1/guidelines/cardiac_arrest?age_group=adult`
- List scenarios: `GET /api/v1/scenarios`

## Core Functions

### `retrieve_guidelines(scenario: str, age_group: str) -> Dict`

Main function for the Care Agent to retrieve guidelines. Returns a dictionary with:
- `text`: Text content of the guidelines
- `images`: List of image file paths
- `metadata`: Metadata dictionary (scenario, age_group, source, etc.)

### `get_available_scenarios() -> List[str]`

Returns a list of all available emergency scenarios in the knowledge base.

### `add_new_scenario(scenario_name: str, ...) -> bool`

Creates directory structure for a new scenario. Placeholder for adding new official guidance.

### `load_knowledge_base() -> Dict`

Loads all knowledge base content into a structured dictionary.

## Adding New Scenarios

### Method 1: Using the function

```python
from retrieval.knowledge_loader import add_new_scenario

add_new_scenario("stroke")
# Creates: knowledge_base/stroke/adult/guidelines.txt
#         knowledge_base/stroke/child/guidelines.txt
```

### Method 2: Manual creation

1. Create directory structure:
```
knowledge_base/
└── new_scenario/
    ├── adult/
    │   ├── guidelines.txt
    │   └── images/
    └── child/
        ├── guidelines.txt
        └── images/
```

2. Add content to `guidelines.txt` files
3. Add images to `images/` directories (optional)

## Integration with Care Agent

The Care Agent should use the `retrieve_guidelines()` function:

```python
from agent_interface import retrieve_guidelines

# In your Care Agent code
def handle_emergency(emergency_type: str, patient_age: str):
    age_group = "adult" if patient_age >= 18 else "child"
    guidelines = retrieve_guidelines(
        scenario=emergency_type,
        age_group=age_group
    )
    
    # Use guidelines["text"] for instructions
    # Use guidelines["images"] for visual aids
    return guidelines
```

## Constraints

- **No embeddings**: This system does not use vector embeddings or semantic search
- **No automatic scraping**: All content must be manually curated
- **Scenario-based only**: Retrieval is based on exact scenario name matching
- **Curated documents only**: All guidelines must be added manually to the knowledge base

## TODO / Future Enhancements

- [ ] Add input validation and error handling
- [ ] Add logging for agent queries
- [ ] Implement file copying in `add_new_scenario()`
- [ ] Add support for metadata.json files
- [ ] Add caching mechanism for performance
- [ ] Add support for scenario aliases
- [ ] Add API authentication if needed
- [ ] Add rate limiting for API endpoints

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

