# Usage Guide

## Quick Start

### Option 1: Direct Import (Recommended)

From the `TrainingAgentScrapingRAG` directory:

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_interface import retrieve_guidelines

# Use the function
guidelines = retrieve_guidelines(scenario="cardiac_arrest", age_group="adult")
print(guidelines["text"])
```

### Option 2: Running as a Module

From the parent directory:

```python
from TrainingAgentScrapingRAG.agent_interface import retrieve_guidelines

guidelines = retrieve_guidelines(scenario="cardiac_arrest", age_group="adult")
```

### Option 3: Using the Demo Script

```bash
cd TrainingAgentScrapingRAG
python main.py
```

## Adding New Scenarios

### Using the Function

```python
from retrieval.knowledge_loader import add_new_scenario

add_new_scenario("stroke")
# Then edit the created files in knowledge_base/stroke/
```

### Manual Creation

1. Create directory: `knowledge_base/new_scenario/adult/` and `knowledge_base/new_scenario/child/`
2. Add `guidelines.txt` to each age group directory
3. Optionally add images to `images/` subdirectories
4. Optionally add `metadata.json` for additional information

## API Usage

Start the FastAPI server:

```bash
cd TrainingAgentScrapingRAG
uvicorn api.endpoints:app --reload
```

Then access:
- API Docs: http://localhost:8000/docs
- Get guidelines: http://localhost:8000/api/v1/guidelines/cardiac_arrest?age_group=adult
- List scenarios: http://localhost:8000/api/v1/scenarios

## Integration with Care Agent

```python
from agent_interface import retrieve_guidelines

def handle_emergency(emergency_type: str, patient_age: int):
    age_group = "adult" if patient_age >= 18 else "child"
    
    guidelines = retrieve_guidelines(
        scenario=emergency_type,
        age_group=age_group
    )
    
    if guidelines["text"]:
        # Provide text guidance
        return guidelines["text"]
    else:
        # Handle scenario not found
        return "Scenario not found. Please contact emergency services."
```

