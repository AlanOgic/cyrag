"""
Embedding model utilities for technical documentation.
"""
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDING_MODEL

class DocumentEmbedder:
    """Embedding generator for technical documentation."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                        If None, uses the model specified in config.
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Set device (use GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        print(f"Initialized embedder with model {self.model_name} on {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of text documents to embed
            batch_size: Number of documents to process at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings"):
            batch_texts = documents[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        return self.model.encode(query, convert_to_numpy=True).tolist()