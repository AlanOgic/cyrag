"""
Qdrant collection management for Cyanview documents.
"""
from typing import List, Dict, Optional, Any
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from config import COLLECTION_NAME
from src.qdrant.client import get_qdrant_client

def create_collection(
    vector_size: int,
    collection_name: Optional[str] = None,
    recreate: bool = False
) -> None:
    """
    Create a new Qdrant collection for document vectors.
    
    Args:
        vector_size: Dimension of embedding vectors
        collection_name: Name of the collection (default from config)
        recreate: Whether to recreate the collection if it exists
    """
    client = get_qdrant_client()
    collection_name = collection_name or COLLECTION_NAME
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        if recreate:
            print(f"Recreating collection '{collection_name}'...")
            client.delete_collection(collection_name=collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. Use recreate=True to recreate it.")
            return
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0  # Index immediately
        )
    )
    
    # Create payload index for filtering
    client.create_payload_index(
        collection_name=collection_name,
        field_name="product",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="type",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    print(f"Created collection '{collection_name}' with vector size {vector_size}")

def store_embeddings(
    vectors: List[List[float]],
    payloads: List[Dict],
    collection_name: Optional[str] = None,
    batch_size: int = 100
) -> None:
    """
    Store embeddings in Qdrant collection.
    
    Args:
        vectors: List of embedding vectors
        payloads: List of payload dictionaries for each vector
        collection_name: Name of the collection (default from config)
        batch_size: Size of batches for uploading
    """
    client = get_qdrant_client()
    collection_name = collection_name or COLLECTION_NAME
    
    # Prepare points
    points = []
    for i, (vector, payload) in enumerate(zip(vectors, payloads)):
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload=payload
        ))
    
    # Upload in batches with progress bar
    for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading vectors to {collection_name}"):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print(f"Uploaded {len(points)} vectors to collection '{collection_name}'")
    
    # Wait for indexing to complete
    time.sleep(1)
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Collection info: {collection_info.status}")

def search_vectors(
    query_vector: List[float],
    collection_name: Optional[str] = None,
    filter_params: Optional[Dict] = None,
    top_k: int = 5,
    score_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Search for similar vectors in Qdrant collection.
    
    Args:
        query_vector: Query vector
        collection_name: Name of the collection (default from config)
        filter_params: Filter conditions
        top_k: Number of results to return
        score_threshold: Minimum similarity score threshold
        
    Returns:
        List of search results with payload and score
    """
    client = get_qdrant_client()
    collection_name = collection_name or COLLECTION_NAME
    
    # Build search parameters
    search_params = {
        "collection_name": collection_name,
        "query_vector": query_vector,
        "limit": top_k
    }
    
    # Add filter if provided
    if filter_params:
        search_params["query_filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
                for key, value in filter_params.items()
            ]
        )
    
    # Add score threshold if provided
    if score_threshold is not None:
        search_params["score_threshold"] = score_threshold
    
    # Execute search
    search_results = client.search(**search_params)
    
    # Format results
    results = []
    for result in search_results:
        results.append({
            "id": result.id,
            "score": result.score,
            "payload": result.payload
        })
    
    return results