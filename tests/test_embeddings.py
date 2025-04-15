"""
Tests for embedding functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.embeddings.embedder import DocumentEmbedder

class TestEmbeddings(unittest.TestCase):
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_embedder_initialization(self, mock_transformer):
        """Test embedder initialization with default model."""
        # Mock transformer instance
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_instance
        
        # Initialize embedder
        embedder = DocumentEmbedder()
        
        # Check that transformer was initialized with expected model
        mock_transformer.assert_called_once()
        
        # Check embedding dimension
        self.assertEqual(embedder.embedding_dim, 768)
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_embed_documents(self, mock_transformer):
        """Test embedding generation for documents."""
        # Mock transformer instance
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        
        # Mock encode method to return mock embeddings
        dummy_embeddings = np.random.random((3, 768))
        mock_instance.encode.return_value = dummy_embeddings
        mock_transformer.return_value = mock_instance
        
        # Initialize embedder
        embedder = DocumentEmbedder()
        
        # Generate embeddings
        docs = ["Document 1", "Document 2", "Document 3"]
        embeddings = embedder.embed_documents(docs)
        
        # Check that encode was called with correct parameters
        mock_instance.encode.assert_called_once_with(docs, convert_to_numpy=True)
        
        # Check embeddings
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(embeddings[0]), 768)
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_embed_query(self, mock_transformer):
        """Test query embedding generation."""
        # Mock transformer instance
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        
        # Mock encode method to return mock embedding
        dummy_embedding = np.random.random(768)
        mock_instance.encode.return_value = dummy_embedding
        mock_transformer.return_value = mock_instance
        
        # Initialize embedder
        embedder = DocumentEmbedder()
        
        # Generate embedding
        query = "How does the RIO connect to cameras?"
        embedding = embedder.embed_query(query)
        
        # Check that encode was called with correct parameters
        mock_instance.encode.assert_called_once_with(query, convert_to_numpy=True)
        
        # Check embedding
        self.assertEqual(len(embedding), 768)

if __name__ == "__main__":
    unittest.main()
