"""
Main Entry Point

Entry point for the healthcare RAG system.
Can be used to run the API server or initialize the RAG pipeline.
"""

import uvicorn
from api.main import app
from config import Config


def main():
    """
    Main entry point for the application.
    
    Starts the FastAPI server for the RAG API.
    """
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info" if not Config.API_DEBUG else "debug"
    )


if __name__ == "__main__":
    main()

