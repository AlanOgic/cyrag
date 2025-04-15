"""
API routes for the Cyanview RAG system.
"""
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.query import rag_query

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="Query text to search for")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    filter_params: Optional[Dict[str, str]] = Field(None, description="Filter parameters")
    score_threshold: Optional[float] = Field(0.7, description="Minimum similarity score")
    is_technical: Optional[bool] = Field(False, description="Whether to use technical prompt")

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str = Field(..., description="Original query text")
    sources: List[str] = Field(..., description="List of document sources")
    answer: Optional[str] = Field(None, description="Generated answer")
    context: Optional[str] = Field(None, description="Retrieved context")

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with the provided text.
    
    Args:
        request: Query request including query text and optional parameters
        
    Returns:
        Query response with answer and sources
    """
    try:
        # Log the request
        logger.info(f"Received query request: {request.query}")
        
        # Execute RAG query
        result = rag_query(
            query_text=request.query,
            top_k=request.top_k,
            filter_params=request.filter_params,
            score_threshold=request.score_threshold,
            is_technical=request.is_technical
        )
        
        # Return response
        return result
    
    except Exception as e:
        # Log the error
        logger.error(f"Error processing query: {e}", exc_info=True)
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )