"""
Text Chunker

Handles splitting text documents into chunks for vector embedding.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_id: str
    source: str
    start_index: int
    end_index: int
    metadata: Dict


class TextChunker:
    """
    Splits text documents into chunks for embedding and retrieval.
    
    Supports multiple chunking strategies optimized for healthcare content.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_strategy: str = "fixed_size"
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            chunking_strategy: Strategy for chunking (fixed_size, semantic, sentence)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
            
        TODO:
            - Implement semantic chunking (preserve medical context)
            - Add sentence-aware chunking
            - Preserve medical procedure steps as complete chunks
            - Handle tables and structured data in medical documents
        """
        if self.chunking_strategy == "fixed_size":
            return self._chunk_fixed_size(text, metadata or {})
        elif self.chunking_strategy == "sentence":
            return self._chunk_by_sentence(text, metadata or {})
        elif self.chunking_strategy == "semantic":
            return self._chunk_semantic(text, metadata or {})
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
    
    def _chunk_fixed_size(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces.
        
        Args:
            text: Text to chunk
            metadata: Metadata for chunks
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunk = Chunk(
                content=chunk_content,
                chunk_id=f"{metadata.get('source_id', 'doc')}_chunk_{chunk_id}",
                source=metadata.get('source', 'unknown'),
                start_index=start,
                end_index=end,
                metadata=metadata.copy()
            )
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def _chunk_by_sentence(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk text by sentences, respecting chunk_size limit.
        
        Args:
            text: Text to chunk
            metadata: Metadata for chunks
            
        Returns:
            List of Chunk objects
            
        TODO:
            - Implement sentence splitting
            - Use medical-aware sentence tokenization
            - Preserve sentence boundaries in medical procedures
        """
        # Placeholder: fall back to fixed_size for now
        return self._chunk_fixed_size(text, metadata)
    
    def _chunk_semantic(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk text semantically, preserving medical context.
        
        Args:
            text: Text to chunk
            metadata: Metadata for chunks
            
        Returns:
            List of Chunk objects
            
        TODO:
            - Implement semantic chunking using embeddings
            - Group related medical concepts together
            - Preserve complete medical procedures
            - Use topic modeling for better chunk boundaries
        """
        # Placeholder: fall back to fixed_size for now
        return self._chunk_fixed_size(text, metadata)
    
    def chunk_batch(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[List[Chunk]]:
        """
        Chunk multiple text documents.
        
        Args:
            texts: List of texts to chunk
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of chunk lists, one per input text
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        return [self.chunk_text(text, meta) for text, meta in zip(texts, metadata_list)]

