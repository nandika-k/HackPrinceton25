# Healthcare RAG System - Cardiac Emergency First-Aid Guide

A modular Retrieval-Augmented Generation (RAG) framework for providing first-aid guidance for cardiac emergencies.

## Project Structure

```
.
├── ingestion/          # Data ingestion modules
│   ├── __init__.py
│   └── ingestor.py    # Data ingestion placeholders
├── preprocessing/      # Text preprocessing and chunking
│   ├── __init__.py
│   ├── preprocessor.py  # Text cleaning and preprocessing
│   └── chunker.py      # Text chunking strategies
├── embedding/          # Vector embedding generation
│   ├── __init__.py
│   └── embedder.py    # Embedding model interface
├── vector_store/       # Vector database operations
│   ├── __init__.py
│   └── vector_db.py   # Vector storage and search
├── retrieval/          # Retrieval logic
│   ├── __init__.py
│   └── retriever.py   # RAG retrieval interface
├── api/                # FastAPI application
│   ├── __init__.py
│   └── main.py        # API endpoints
├── config.py           # Configuration settings
├── pipeline.py         # Complete RAG pipeline
├── main.py             # Entry point for API server
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Features

- **Modular Architecture**: Separated concerns for easy integration and modification
- **Placeholder Methods**: All core methods are stubbed with clear TODOs
- **FastAPI Integration**: Ready-to-use API endpoints for the Care Agent
- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Comprehensive docstrings and comments

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file:

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
CHUNK_SIZE=500
CHUNK_OVERLAP=50
VECTOR_DB_TYPE=faiss
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4
LLM_API_KEY=your_api_key_here
```

### 3. Run the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

### Retrieve Guidelines
```
POST /guidelines
Body: {
    "condition": "cardiac arrest",
    "top_k": 5
}
```

### General Retrieval
```
POST /retrieve
Body: {
    "query": "What to do for chest pain?",
    "top_k": 5
}
```

### RAG Generation
```
POST /rag/generate
Body: {
    "query": "How to perform CPR?",
    "condition": "cardiac arrest"
}
```

### Database Statistics
```
GET /stats
```

## Usage

### Running the Pipeline

To ingest and index healthcare data:

```python
from pipeline import RAGPipeline

pipeline = RAGPipeline()
data_sources = ["data/guidelines.txt"]
pipeline.run_pipeline(data_sources)
```

### Using the Retriever

```python
from retrieval.retriever import Retriever
from embedding.embedder import Embedder
from vector_store.vector_db import VectorDB

embedder = Embedder()
vector_db = VectorDB()
retriever = Retriever(embedder, vector_db)

guidelines = retriever.get_guidelines("cardiac arrest", top_k=5)
```

## Implementation TODOs

### Data Ingestion
- [ ] Implement actual data reading from files (PDF, TXT, JSON)
- [ ] Add support for medical databases
- [ ] Implement data validation and sanitization

### Preprocessing
- [ ] Add medical NER (Named Entity Recognition)
- [ ] Implement medical term normalization
- [ ] Add sentence-aware chunking

### Embeddings
- [ ] Integrate actual embedding models (sentence-transformers, OpenAI)
- [ ] Add support for medical-specific embedding models
- [ ] Implement embedding caching

### Vector Store
- [ ] Integrate actual vector database (FAISS, Pinecone, ChromaDB)
- [ ] Implement index persistence
- [ ] Add metadata filtering
- [ ] Implement hybrid search

### Retrieval
- [ ] Add query expansion for medical terms
- [ ] Implement reranking
- [ ] Add support for multi-query retrieval

### LLM Integration
- [ ] Integrate with LLM provider (OpenAI, Anthropic)
- [ ] Format prompts with retrieved context
- [ ] Add citation tracking
- [ ] Implement streaming responses

## Integration with Care Agent

The main endpoint for the Care Agent is:

```python
POST /guidelines
{
    "condition": "cardiac arrest",
    "top_k": 5
}
```

This returns relevant first-aid guidelines that can be used by the Care Agent to provide guidance to users.

## Development

### Adding New Data Sources

1. Extend `DataIngestor` in `ingestion/ingestor.py`
2. Implement the specific ingestion method
3. Update the pipeline to use the new source

### Adding New Chunking Strategies

1. Extend `TextChunker` in `preprocessing/chunker.py`
2. Implement the new chunking method
3. Update configuration to use the new strategy

### Customizing Embeddings

1. Update `Embedder` in `embedding/embedder.py`
2. Load your preferred embedding model
3. Update configuration with model details

## License

This project is created for HackPrinceton 2025.

## Notes

- All methods are placeholders and require actual implementation
- The system is designed for easy integration with healthcare data
- Focus on cardiac emergencies, but framework is extensible to other conditions
- All TODOs are marked in the code for easy identification

