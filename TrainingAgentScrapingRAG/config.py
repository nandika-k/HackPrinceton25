"""
Configuration Module

Central configuration for the RAG system.
"""

from typing import Dict, Any
from pathlib import Path
import os


class Config:
    """Configuration class for RAG system settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    INDEX_DIR = PROJECT_ROOT / "indices"
    
    # Embedding configuration
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Chunking configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed_size")  # fixed_size, sentence, semantic
    
    # Vector database configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # faiss, pinecone, chroma, weaviate
    VECTOR_DB_INDEX_PATH = os.getenv("VECTOR_DB_INDEX_PATH", str(INDEX_DIR / "vector_index"))
    
    # Retrieval configuration
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
    
    # LLM configuration (for RAG generation)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, local
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Data ingestion configuration
    DATA_SOURCES = os.getenv("DATA_SOURCES", "").split(",") if os.getenv("DATA_SOURCES") else []
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.INDEX_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "embedding_dim": cls.EMBEDDING_DIM,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "vector_db_type": cls.VECTOR_DB_TYPE,
            "default_top_k": cls.DEFAULT_TOP_K,
            "llm_provider": cls.LLM_PROVIDER,
            "llm_model": cls.LLM_MODEL_NAME
        }


# Create directories on import
Config.create_directories()

