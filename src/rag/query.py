"""
RAG query processing for Cyanview documentation.
"""
from typing import Dict, List, Optional, Any, Union
import os
import logging

from langchain.llms import OpenAI
from langchain.chains import LLMChain

from config import COLLECTION_NAME, OPENAI_API_KEY, LLM_PROVIDER
from src.embeddings.embedder import DocumentEmbedder
from src.qdrant.collection import search_vectors
from src.rag.prompt_templates import get_rag_prompt, get_technical_rag_prompt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize embedder
embedder = DocumentEmbedder()

def format_context(results: List[Dict]) -> str:
    """
    Format retrieved documents into context string.
    
    Args:
        results: List of retrieved documents
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(results):
        # Format with source information
        source = result.get("source", "").split("/")[-1] if result.get("source") else f"Document {i+1}"
        product = f" [{result.get('product', '')}]" if result.get("product") else ""
        
        # Format the text with source reference
        context_part = f"[Source: {source}{product}, Score: {result.get('score', 0):.2f}]\n{result.get('text', '')}\n\n"
        context_parts.append(context_part)
    
    return "\n".join(context_parts)

def query_qdrant(
    query_text: str, 
    top_k: int = 5,
    filter_params: Optional[Dict] = None,
    score_threshold: float = 0.7
) -> List[Dict]:
    """
    Query the Qdrant collection for relevant documents.
    
    Args:
        query_text: Query text
        top_k: Number of results to return
        filter_params: Optional filter parameters
        score_threshold: Minimum similarity score threshold
        
    Returns:
        List of relevant documents with metadata
    """
    # Generate embedding for the query
    query_vector = embedder.embed_query(query_text)
    
    # Search Qdrant
    search_results = search_vectors(
        query_vector=query_vector,
        collection_name=COLLECTION_NAME,
        filter_params=filter_params,
        top_k=top_k,
        score_threshold=score_threshold
    )
    
    # Format results
    contexts = []
    for result in search_results:
        if "text" in result["payload"]:
            contexts.append({
                "text": result["payload"]["text"],
                "source": result["payload"].get("source", ""),
                "product": result["payload"].get("product", ""),
                "score": result["score"],
                "id": result["id"]
            })
    
    return contexts

def get_llm_chain(is_technical: bool = False):
    """
    Get a configured LLM chain based on environment settings.
    
    Args:
        is_technical: Whether to use the technical prompt template
    
    Returns:
        Configured LLM chain
    """
    if LLM_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key provided. Response generation disabled.")
            return None
        
        # Initialize OpenAI LLM
        llm = OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.1,  # Low temperature for more factual responses
            max_tokens=1000   # Reasonable response length
        )
        
        # Get appropriate prompt template
        prompt = get_technical_rag_prompt() if is_technical else get_rag_prompt()
        
        # Create and return chain
        return LLMChain(llm=llm, prompt=prompt)
    
    else:
        logger.warning(f"Unsupported LLM provider: {LLM_PROVIDER}")
        return None

def rag_query(
    query_text: str,
    top_k: int = 5,
    filter_params: Optional[Dict] = None,
    score_threshold: float = 0.7,
    is_technical: bool = False
) -> Dict:
    """
    Execute a RAG query against the Cyanview documentation.
    
    Args:
        query_text: Query text
        top_k: Number of results to return
        filter_params: Optional filter parameters (e.g., {"product": "RCP"})
        score_threshold: Minimum similarity score threshold
        is_technical: Whether to use the technical prompt template
    
    Returns:
        Dictionary with query results including context, sources, and answer if available
    """
    # Log the query
    logger.info(f"Processing query: {query_text}")
    
    # Query for relevant documents
    search_results = query_qdrant(
        query_text=query_text,
        top_k=top_k,
        filter_params=filter_params,
        score_threshold=score_threshold
    )
    
    # Check if any results were found
    if not search_results:
        logger.warning("No relevant documents found for the query.")
        return {
            "query": query_text,
            "context": "",
            "sources": [],
            "answer": "I couldn't find any relevant information to answer your question about Cyanview systems."
        }
    
    # Format context from results
    context = format_context(search_results)
    
    # Extract sources for reference
    sources = [result.get("source", "Unknown source") for result in search_results]
    
    # Initialize response
    response = {
        "query": query_text,
        "context": context,
        "sources": sources
    }
    
    # Generate answer if LLM is configured
    llm_chain = get_llm_chain(is_technical=is_technical)
    if llm_chain:
        try:
            answer = llm_chain.run(context=context, query=query_text)
            response["answer"] = answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            response["answer"] = "Error generating answer based on the retrieved context."
    
    return response