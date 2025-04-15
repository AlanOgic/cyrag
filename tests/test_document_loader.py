"""
Tests for document loading functionality.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

from src.data.document_loader import load_documents
from src.data.json_loader import load_json_qa

class TestDocumentLoader(unittest.TestCase):
    def test_load_json_qa_list_format(self):
        """Test loading JSON file with list of Q&A pairs."""
        test_json = '''[
            {
                "question": "What is the RCP?",
                "answer": "The RCP is Cyanview's Remote Control Panel.",
                "metadata": {"product": "RCP", "topic": "Core Functionality"}
            },
            {
                "question": "How does RIO connect to cameras?",
                "response": "RIO connects to cameras via IP, serial, or USB.",
                "metadata": {"product": "RIO", "topic": "Connectivity"}
            }
        ]'''
        
        with patch("builtins.open", mock_open(read_data=test_json)):
            docs = load_json_qa("dummy_path.json")
        
        self.assertEqual(len(docs), 2)
        self.assertIn("Q: What is the RCP?", docs[0].page_content)
        self.assertIn("A: The RCP is Cyanview's Remote Control Panel.", docs[0].page_content)
        self.assertIn("Q: How does RIO connect to cameras?", docs[1].page_content)
        self.assertIn("A: RIO connects to cameras via IP, serial, or USB.", docs[1].page_content)
        
        # Check metadata
        self.assertEqual(docs[0].metadata.get("product"), "RCP")
        self.assertEqual(docs[0].metadata.get("topic"), "Core Functionality")
        self.assertEqual(docs[1].metadata.get("product"), "RIO")
        self.assertEqual(docs[1].metadata.get("topic"), "Connectivity")
    
    def test_load_json_qa_dict_format(self):
        """Test loading JSON file with nested FAQ structure."""
        test_json = '''{
            "faq": [
                {
                    "question": "What is the CI0?",
                    "answer": "The CI0 is a Camera Interface for integrating serial cameras."
                },
                {
                    "question": "What is the VP4?",
                    "answer": "The VP4 is a Video Processor for color correction."
                }
            ]
        }'''
        
        with patch("builtins.open", mock_open(read_data=test_json)):
            docs = load_json_qa("dummy_path.json")
        
        self.assertEqual(len(docs), 2)
        self.assertIn("Q: What is the CI0?", docs[0].page_content)
        self.assertIn("A: The CI0 is a Camera Interface for integrating serial cameras.", docs[0].page_content)
        self.assertIn("Q: What is the VP4?", docs[1].page_content)
        self.assertIn("A: The VP4 is a Video Processor for color correction.", docs[1].page_content)

if __name__ == "__main__":
    unittest.main()
