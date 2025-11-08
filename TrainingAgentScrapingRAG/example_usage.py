"""
Example Usage Script

Demonstrates how to use the RAG system components.
"""

from ingestion.ingestor import DataIngestor
from preprocessing.preprocessor import TextPreprocessor
from preprocessing.chunker import TextChunker
from embedding.embedder import Embedder
from vector_store.vector_db import VectorDB
from retrieval.retriever import Retriever
from pipeline import RAGPipeline


def example_basic_usage():
    """Example of basic RAG pipeline usage."""
    print("=== Example: Basic RAG Pipeline Usage ===\n")
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Run pipeline with placeholder data sources
    # In actual implementation, provide real data source paths
    data_sources = ["data/guidelines.txt"]
    print("Running RAG pipeline...")
    pipeline.run_pipeline(data_sources)
    print("\n")


def example_retriever_usage():
    """Example of using the retriever directly."""
    print("=== Example: Direct Retriever Usage ===\n")
    
    # Initialize components
    embedder = Embedder()
    vector_db = VectorDB()
    retriever = Retriever(embedder, vector_db)
    
    # Example: Retrieve guidelines for a condition
    condition = "cardiac arrest"
    print(f"Retrieving guidelines for: {condition}")
    guidelines = retriever.get_guidelines(condition, top_k=5)
    
    print(f"Found {len(guidelines)} guidelines:")
    for i, guideline in enumerate(guidelines, 1):
        print(f"\n{i}. Score: {guideline['score']:.3f}")
        print(f"   Content: {guideline['content'][:100]}...")
    print("\n")


def example_custom_pipeline():
    """Example of building a custom pipeline."""
    print("=== Example: Custom Pipeline ===\n")
    
    # Initialize individual components
    ingestor = DataIngestor()
    preprocessor = TextPreprocessor(lowercase=False)
    chunker = TextChunker(chunk_size=300, chunk_overlap=30)
    embedder = Embedder(embedding_dim=384)
    vector_db = VectorDB(db_type="faiss")
    
    # Ingest data
    print("Ingesting data...")
    ingested = ingestor.ingest_data("example_source.txt")
    
    # Preprocess
    print("Preprocessing...")
    texts = [item["content"] for item in ingested]
    preprocessed = preprocessor.preprocess_batch(texts)
    
    # Chunk
    print("Chunking...")
    chunks_list = chunker.chunk_batch(preprocessed)
    
    # Embed and store
    print("Embedding and storing...")
    for chunks, item in zip(chunks_list, ingested):
        chunk_contents = [chunk.content for chunk in chunks]
        embeddings = embedder.embed_chunks(chunk_contents)
        metadata = [chunk.metadata for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        vector_db.store_embeddings(
            embeddings=embeddings,
            chunks=chunk_contents,
            metadata=metadata,
            chunk_ids=chunk_ids
        )
    
    # Create retriever and search
    retriever = Retriever(embedder, vector_db)
    results = retriever.retrieve("chest pain", top_k=3)
    
    print(f"Retrieved {len(results)} results")
    print("\n")


def example_api_usage():
    """Example of API endpoint usage."""
    print("=== Example: API Endpoint Usage ===\n")
    
    print("To use the API, start the server:")
    print("  python main.py")
    print("\nThen make HTTP requests:")
    print("\n1. Health check:")
    print("   GET http://localhost:8000/health")
    print("\n2. Get guidelines:")
    print("   POST http://localhost:8000/guidelines")
    print("   Body: {\"condition\": \"cardiac arrest\", \"top_k\": 5}")
    print("\n3. Retrieve chunks:")
    print("   POST http://localhost:8000/retrieve")
    print("   Body: {\"query\": \"chest pain\", \"top_k\": 5}")
    print("\n4. RAG generation:")
    print("   POST http://localhost:8000/rag/generate")
    print("   Body: {\"query\": \"How to perform CPR?\", \"condition\": \"cardiac arrest\"}")
    print("\n")


if __name__ == "__main__":
    print("Healthcare RAG System - Example Usage\n")
    print("=" * 50)
    print("\n")
    
    # Note: These examples use placeholder implementations
    # Actual data and models need to be integrated
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Error in basic usage example: {e}\n")
    
    try:
        example_retriever_usage()
    except Exception as e:
        print(f"Error in retriever usage example: {e}\n")
    
    try:
        example_custom_pipeline()
    except Exception as e:
        print(f"Error in custom pipeline example: {e}\n")
    
    example_api_usage()
    
    print("=" * 50)
    print("\nNote: These are placeholder examples.")
    print("Actual implementation requires:")
    print("  - Real data sources")
    print("  - Actual embedding models")
    print("  - Vector database setup")
    print("  - LLM integration")

