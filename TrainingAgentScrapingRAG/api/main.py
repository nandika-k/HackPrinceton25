"""
FastAPI Application

Main API endpoints for the healthcare RAG system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from retrieval.retriever import Retriever
from embedding.embedder import Embedder
from vector_store.vector_db import VectorDB

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare RAG API",
    description="RAG system API for cardiac emergency first-aid guidance",
    version="1.0.0"
)

# CORS middleware for hackathon integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components (placeholder - will be initialized properly in actual implementation)
embedder = Embedder()
vector_db = VectorDB()
retriever = Retriever(embedder, vector_db)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for retrieval queries."""
    query: str = Field(..., description="Query string for retrieval")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class ConditionRequest(BaseModel):
    """Request model for condition-based guideline retrieval."""
    condition: str = Field(..., description="Medical condition or emergency type")
    top_k: int = Field(5, ge=1, le=20, description="Number of guideline chunks to return")


class GuidelineResponse(BaseModel):
    """Response model for guideline retrieval."""
    content: str = Field(..., description="Guideline text content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    chunk_id: str = Field(..., description="Unique chunk identifier")


class GuidelinesResponse(BaseModel):
    """Response model for multiple guidelines."""
    condition: str = Field(..., description="Requested condition")
    guidelines: List[GuidelineResponse] = Field(..., description="List of retrieved guidelines")
    total_results: int = Field(..., description="Total number of results")


class RAGRequest(BaseModel):
    """Request model for RAG generation."""
    query: str = Field(..., description="User query")
    condition: Optional[str] = Field(None, description="Optional condition for filtering")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


class RAGResponse(BaseModel):
    """Response model for RAG generation."""
    response: str = Field(..., description="Generated response")
    query: str = Field(..., description="Original query")
    context_length: int = Field(..., description="Length of retrieved context")
    condition: Optional[str] = Field(None, description="Condition used for filtering")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")


# API Endpoints
@app.get("/", response_model=HealthCheckResponse)
async def root():
    """
    Root endpoint for health check.
    
    Returns:
        Health check response with API status
    """
    return HealthCheckResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health check response
    """
    # TODO: Add actual health checks (database connectivity, model loading, etc.)
    return HealthCheckResponse(status="healthy", version="1.0.0")


@app.post("/retrieve", response_model=List[GuidelineResponse])
async def retrieve(request: QueryRequest):
    """
    Retrieve relevant chunks for a query.
    
    Args:
        request: QueryRequest with query string and parameters
        
    Returns:
        List of relevant guideline chunks
        
    TODO:
        - Add query validation
        - Add error handling for retrieval failures
        - Add logging
    """
    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )
        
        return [
            GuidelineResponse(
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id
            )
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.post("/guidelines", response_model=GuidelinesResponse)
async def get_guidelines(request: ConditionRequest):
    """
    Get first-aid guidelines for a specific condition.
    
    This is the main endpoint that the Care Agent will call.
    
    Args:
        request: ConditionRequest with condition name and parameters
        
    Returns:
        GuidelinesResponse with retrieved guidelines
        
    TODO:
        - Add condition validation
        - Add support for multiple conditions
        - Add caching for frequently requested conditions
        - Integrate with LLM for response generation
    """
    try:
        guidelines = retriever.get_guidelines(
            condition=request.condition,
            top_k=request.top_k
        )
        
        return GuidelinesResponse(
            condition=request.condition,
            guidelines=[
                GuidelineResponse(**guideline)
                for guideline in guidelines
            ],
            total_results=len(guidelines)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve guidelines: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the vector database.
    
    Returns:
        Dictionary with database statistics
        
    TODO:
        - Add more detailed statistics
        - Add performance metrics
    """
    try:
        stats = vector_db.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# RAG Integration Point
# This is where the vector store connects to a generator (LLM)
@app.post("/rag/generate", response_model=RAGResponse)
async def generate_response(request: RAGRequest):
    """
    Generate a response using RAG (Retrieval-Augmented Generation).
    
    This endpoint combines retrieval with LLM generation:
    1. Retrieve relevant context using the retriever
    2. Format context for LLM prompt
    3. Generate response using LLM
    4. Return formatted response
    
    Args:
        request: RAGRequest with query and optional condition
        
    Returns:
        RAGResponse with generated response and metadata
        
    TODO:
        - Integrate with LLM (OpenAI, Anthropic, local model)
        - Format prompt with retrieved context
        - Generate and return response
        - Add citation tracking
        - Add streaming support for long responses
    """
    try:
        # Step 1: Retrieve relevant context
        if request.condition:
            # Use condition-specific retrieval
            results = retriever.retrieve(
                query=request.condition,
                top_k=request.top_k
            )
        else:
            # Use query-based retrieval
            results = retriever.retrieve(
                query=request.query,
                top_k=request.top_k
            )
        
        # Step 2: Format context for LLM
        context = retriever.format_context_for_llm(results)
        
        # Step 3: Format prompt (placeholder)
        prompt = f"""You are a healthcare AI assistant providing first-aid guidance for cardiac emergencies.

Context:
{context}

User Query: {request.query}

Please provide accurate, helpful first-aid guidance based on the context above.
"""
        
        # Step 4: Generate response (placeholder)
        # In actual implementation, this would call an LLM
        # Example: response = llm.generate(prompt)
        response = f"[LLM Response Placeholder]\nQuery: {request.query}\nContext retrieved: {len(context)} characters\n\nBased on the retrieved context, provide first-aid guidance here."
        
        return RAGResponse(
            response=response,
            query=request.query,
            context_length=len(context),
            condition=request.condition
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


# Export router for potential use in other modules
router = app

