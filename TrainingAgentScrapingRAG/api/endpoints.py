"""
FastAPI endpoints for first-aid guidelines retrieval.

This module provides REST API endpoints that expose the retrieval
functionality via HTTP. This is optional and can be used if the
Care Agent needs to access guidelines via API calls.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent_interface.care_agent import retrieve_guidelines, get_available_scenarios

# Initialize FastAPI app
app = FastAPI(
    title="First-Aid Guidelines API",
    description="API for retrieving curated first-aid guidelines by scenario and age group",
    version="1.0.0"
)

# Create router (optional, for better organization)
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1", tags=["guidelines"])


@router.get("/guidelines/{scenario}")
async def get_guidelines_endpoint(
    scenario: str,
    age_group: str = Query(default="adult", regex="^(adult|child)$"),
    knowledge_base_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve first-aid guidelines for a specific scenario and age group.
    
    Args:
        scenario: Name of the emergency scenario (e.g., "cardiac_arrest", "choking").
        age_group: Age group for the guidelines ("adult" or "child"). Defaults to "adult".
        knowledge_base_path: Optional path to knowledge base (for testing).
    
    Returns:
        JSON response with guidelines containing text, images, and metadata.
    
    Example:
        GET /api/v1/guidelines/cardiac_arrest?age_group=adult
    """
    try:
        base_path = Path(knowledge_base_path) if knowledge_base_path else None
        guidelines = retrieve_guidelines(
            scenario=scenario,
            age_group=age_group,
            knowledge_base_path=base_path
        )
        
        # Check if scenario was found
        if not guidelines.get("text") and "error" in guidelines.get("metadata", {}):
            raise HTTPException(
                status_code=404,
                detail=f"Scenario '{scenario}' with age_group '{age_group}' not found"
            )
        
        return JSONResponse(content=guidelines)
    
    except Exception as e:
        # TODO: Add proper error handling and logging
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios")
async def list_scenarios_endpoint(
    knowledge_base_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    List all available emergency scenarios.
    
    Args:
        knowledge_base_path: Optional path to knowledge base (for testing).
    
    Returns:
        JSON response with list of available scenarios.
    
    Example:
        GET /api/v1/scenarios
    """
    try:
        base_path = Path(knowledge_base_path) if knowledge_base_path else None
        scenarios = get_available_scenarios(knowledge_base_path=base_path)
        
        return JSONResponse(content={"scenarios": scenarios})
    
    except Exception as e:
        # TODO: Add proper error handling and logging
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        JSON response indicating API is running.
    """
    return JSONResponse(content={"status": "healthy", "service": "first-aid-guidelines-api"})


# Include router in app
app.include_router(router)


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.
    """
    return {
        "message": "First-Aid Guidelines API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# TODO: Add authentication/authorization if needed
# TODO: Add rate limiting middleware
# TODO: Add request logging
# TODO: Add CORS configuration if needed
# TODO: Add OpenAPI schema customization

