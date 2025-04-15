"""
API server for the Cyanview RAG system.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import router
from config import COLLECTION_NAME

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cyanview RAG API",
    description="API for querying Cyanview technical documentation using Retrieval-Augmented Generation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add routes
app.include_router(router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred", "details": str(exc)},
    )

@app.get("/")
async def root():
    return {
        "message": "Cyanview RAG API",
        "description": "API for querying Cyanview technical documentation",
        "endpoints": [
            {"path": "/query", "method": "POST", "description": "Query the RAG system"}
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "collection": COLLECTION_NAME}

def run_server():
    """Run the API server."""
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_server()