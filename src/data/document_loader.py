"""
Document loading utilities for various file formats.
"""
import os
from typing import List, Optional, Dict, Any

from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

from src.data.json_loader import load_json_qa
from src.utils.metadata import extract_metadata_from_content

def load_documents(directory_path: str) -> List[Document]:
    """
    Load documents from various file formats in the specified directory.
    
    Args:
        directory_path: Path to the directory containing documents.
        
    Returns:
        List of Document objects.
    """
    documents = []
    
    # Walk through directory
    for root, _, files in os.walk(directory_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Process based on file extension
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(filepath)
                    docs = loader.load()
                elif filename.endswith('.txt'):
                    loader = TextLoader(filepath)
                    docs = loader.load()
                elif filename.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(filepath)
                    docs = loader.load()
                elif filename.endswith('.json'):
                    docs = load_json_qa(filepath)
                else:
                    # Skip unsupported file types
                    continue
                
                # Add source metadata if not present
                for doc in docs:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = filepath
                    
                    # Extract additional metadata from content
                    additional_metadata = extract_metadata_from_content(doc.page_content)
                    doc.metadata.update(additional_metadata)
                
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {filepath}")
            
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return documents