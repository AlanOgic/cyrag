"""
Qdrant client for vector storage.
"""
from typing import Optional, Dict, List, Any, Union
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY

class QdrantClientManager:
    """Manager for Qdrant client connection."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantClientManager, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance
    
    def _init_client(self):
        """Initialize the Qdrant client."""
        # Check if URL is a cloud URL or local connection
        if QDRANT_URL.startswith(('http://', 'https://')):
            # Cloud connection
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            self.connection_type = "cloud"
        else:
            # Local connection
            self.client = QdrantClient(
                host=QDRANT_URL,
                port=QDRANT_PORT
            )
            self.connection_type = "local"
        
        # Test connection
        try:
            self.client.get_collections()
            print(f"Successfully connected to Qdrant ({self.connection_type})")
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            raise
    
    def get_client(self) -> QdrantClient:
        """Get the initialized Qdrant client."""
        return self.client

def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client instance."""
    manager = QdrantClientManager()
    return manager.get_client()