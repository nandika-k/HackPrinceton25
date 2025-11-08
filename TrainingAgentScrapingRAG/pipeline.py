"""
RAG Pipeline

Complete pipeline for ingesting, processing, and indexing healthcare data.
"""

from typing import List, Dict
from pathlib import Path
from ingestion.ingestor import DataIngestor
from preprocessing.preprocessor import TextPreprocessor
from preprocessing.chunker import TextChunker
from embedding.embedder import Embedder
from vector_store.vector_db import VectorDB
from config import Config


class RAGPipeline:
    """
    Complete RAG pipeline for healthcare data.
    
    Orchestrates the entire process from data ingestion to vector storage.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with all components."""
        self.ingestor = DataIngestor()
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            chunking_strategy=Config.CHUNKING_STRATEGY
        )
        self.embedder = Embedder(
            model_name=Config.EMBEDDING_MODEL_NAME,
            embedding_dim=Config.EMBEDDING_DIM,
            batch_size=Config.EMBEDDING_BATCH_SIZE
        )
        self.vector_db = VectorDB(
            db_type=Config.VECTOR_DB_TYPE,
            index_path=Config.VECTOR_DB_INDEX_PATH
        )
    
    def run_pipeline(self, data_sources: List[str]) -> None:
        """
        Run the complete RAG pipeline on data sources.
        
        Steps:
        1. Ingest data from sources
        2. Preprocess text
        3. Chunk documents
        4. Generate embeddings
        5. Store in vector database
        
        Args:
            data_sources: List of data source paths or identifiers
            
        TODO:
            - Add progress tracking
            - Add error handling and recovery
            - Add logging
            - Add batch processing for large datasets
        """
        print("Starting RAG pipeline...")
        
        # Step 1: Ingest data
        print("Step 1: Ingesting data...")
        all_ingested_data = []
        for source in data_sources:
            ingested = self.ingestor.ingest_data(source)
            all_ingested_data.extend(ingested)
        print(f"Ingested {len(all_ingested_data)} items")
        
        # Step 2: Preprocess text
        print("Step 2: Preprocessing text...")
        texts = [item["content"] for item in all_ingested_data]
        preprocessed_texts = self.preprocessor.preprocess_batch(texts)
        print(f"Preprocessed {len(preprocessed_texts)} texts")
        
        # Step 3: Chunk documents
        print("Step 3: Chunking documents...")
        all_chunks = []
        all_metadata = []
        all_chunk_ids = []
        
        for i, (text, item) in enumerate(zip(preprocessed_texts, all_ingested_data)):
            chunks = self.chunker.chunk_text(text, item.get("metadata", {}))
            all_chunks.extend([chunk.content for chunk in chunks])
            all_metadata.extend([chunk.metadata for chunk in chunks])
            all_chunk_ids.extend([chunk.chunk_id for chunk in chunks])
        print(f"Created {len(all_chunks)} chunks")
        
        # Step 4: Generate embeddings
        print("Step 4: Generating embeddings...")
        embeddings = self.embedder.embed_chunks(all_chunks)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Step 5: Store in vector database
        print("Step 5: Storing in vector database...")
        self.vector_db.store_embeddings(
            embeddings=embeddings,
            chunks=all_chunks,
            metadata=all_metadata,
            chunk_ids=all_chunk_ids
        )
        print("Pipeline complete!")
        
        # Print statistics
        stats = self.vector_db.get_stats()
        print(f"Vector DB stats: {stats}")
    
    def load_existing_index(self, index_path: str) -> None:
        """
        Load an existing vector index.
        
        Args:
            index_path: Path to the index file
        """
        self.vector_db.load_index(index_path)
        print(f"Loaded index from {index_path}")
    
    def save_index(self, index_path: str) -> None:
        """
        Save the vector index to disk.
        
        Args:
            index_path: Path to save the index
        """
        self.vector_db.save_index(index_path)
        print(f"Saved index to {index_path}")


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()
    
    # Run pipeline with placeholder data sources
    # In actual implementation, provide real data source paths
    data_sources = ["data/guidelines.txt", "data/first_aid_manual.pdf"]
    pipeline.run_pipeline(data_sources)

