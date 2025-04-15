#!/usr/bin/env python
"""
Script to ingest documents into the Cyanview RAG system.
"""
import argparse
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import the local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.document_loader import load_documents
from src.data.text_splitter import split_documents
from src.embeddings.embedder import DocumentEmbedder
from src.qdrant.collection import create_collection, store_embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_documents(docs_dir: str, recreate: bool = False):
    """
    Process documents from directory and store in Qdrant.
    
    Args:
        docs_dir: Path to directory containing documents
        recreate: Whether to recreate the collection if it exists
    """
    start_time = time.time()
    
    # Step 1: Load documents
    logger.info(f"Loading documents from {docs_dir}")
    documents = load_documents(docs_dir)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Split documents into chunks
    logger.info(f"Splitting documents with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info(f"Created {len(chunks)} document chunks")
    
    # Step 3: Initialize embedder
    embedder = DocumentEmbedder()
    
    # Step 4: Create collection
    logger.info("Creating Qdrant collection")
    create_collection(vector_size=embedder.embedding_dim, recreate=recreate)
    
    # Step 5: Generate embeddings
    logger.info("Generating embeddings")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Step 6: Prepare payloads
    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", f"chunk_{i}"),
            "chunk": chunk.metadata.get("chunk", 0),
            "chunk_total": chunk.metadata.get("chunk_total", 1)
        }
        
        # Include additional metadata
        for key, value in chunk.metadata.items():
            if key not in payload and key not in ["page_content"]:
                payload[key] = value
        
        payloads.append(payload)
    
    # Step 7: Store embeddings
    logger.info("Storing embeddings in Qdrant")
    store_embeddings(vectors=embeddings, payloads=payloads)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Cyanview RAG system")
    parser.add_argument("--docs_dir", type=str, required=True, help="Path to directory containing documents")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    
    args = parser.parse_args()
    
    process_documents(args.docs_dir, args.recreate)

if __name__ == "__main__":
    main()
