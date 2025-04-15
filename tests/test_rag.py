"""
Tests for the RAG query system.
"""
import unittest
from unittest.mock import patch, MagicMock

from src.rag.query import query_qdrant, format_context, rag_query

class TestRagSystem(unittest.TestCase):
    @patch('src.rag.query.search_vectors')
    @patch('src.rag.query.embedder')
    def test_query_qdrant(self, mock_embedder, mock_search_vectors):
        """Test querying Qdrant for documents."""
        # Mock embedder
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock search results
        mock_results = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "text": "This is document 1",
                    "source": "doc1.txt",
                    "product": "RCP"
                }
            },
            {
                "id": 2,
                "score": 0.85,
                "payload": {
                    "text": "This is document 2",
                    "source": "doc2.txt",
                    "product": "RIO"
                }
            }
        ]
        mock_search_vectors.return_value = mock_results
        
        # Execute query
        results = query_qdrant("How does RCP work?", top_k=2)
        
        # Check that embed_query was called
        mock_embedder.embed_query.assert_called_once_with("How does RCP work?")
        
        # Check that search_vectors was called with correct parameters
        mock_search_vectors.assert_called_once()
        args, kwargs = mock_search_vectors.call_args
        self.assertEqual(kwargs["query_vector"], [0.1, 0.2, 0.3])
        self.assertEqual(kwargs["top_k"], 2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "This is document 1")
        self.assertEqual(results[0]["product"], "RCP")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[1]["text"], "This is document 2")
        self.assertEqual(results[1]["product"], "RIO")
        self.assertEqual(results[1]["score"], 0.85)
    
    def test_format_context(self):
        """Test formatting context from retrieved documents."""
        results = [
            {
                "text": "The RCP is Cyanview's Remote Control Panel.",
                "source": "doc1.txt",
                "product": "RCP",
                "score": 0.95
            },
            {
                "text": "The RIO enables remote production over the internet.",
                "source": "doc2.txt",
                "product": "RIO",
                "score": 0.85
            }
        ]
        
        context = format_context(results)
        
        # Check that context contains document texts
        self.assertIn("The RCP is Cyanview's Remote Control Panel.", context)
        self.assertIn("The RIO enables remote production over the internet.", context)
        
        # Check that context includes source and product information
        self.assertIn("Source: doc1.txt [RCP]", context)
        self.assertIn("Source: doc2.txt [RIO]", context)
    
    @patch('src.rag.query.query_qdrant')
    @patch('src.rag.query.get_llm_chain')
    def test_rag_query(self, mock_get_llm_chain, mock_query_qdrant):
        """Test full RAG query processing."""
        # Mock query_qdrant results
        mock_query_qdrant.return_value = [
            {
                "text": "The RCP is Cyanview's Remote Control Panel.",
                "source": "doc1.txt",
                "product": "RCP",
                "score": 0.95
            }
        ]
        
        # Mock LLM chain
        mock_chain = MagicMock()
        mock_chain.run.return_value = "The RCP is Cyanview's central control interface."
        mock_get_llm_chain.return_value = mock_chain
        
        # Execute query
        result = rag_query("What is the RCP?", top_k=1)
        
        # Check that query_qdrant was called
        mock_query_qdrant.assert_called_once_with(
            query_text="What is the RCP?", 
            top_k=1, 
            filter_params=None, 
            score_threshold=0.7
        )
        
        # Check that LLM chain was created and run
        mock_get_llm_chain.assert_called_once_with(is_technical=False)
        mock_chain.run.assert_called_once()
        
        # Check result structure
        self.assertEqual(result["query"], "What is the RCP?")
        self.assertIn("doc1.txt", result["sources"])
        self.assertEqual(result["answer"], "The RCP is Cyanview's central control interface.")

if __name__ == "__main__":
    unittest.main()
