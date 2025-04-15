#!/usr/bin/env python
"""
Script to fine-tune the embedder model for Cyanview-specific data.
"""
import argparse
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import the local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.document_loader import load_documents
from src.embeddings.fine_tuning import fine_tune_embedder, prepare_training_examples
from config import DATA_DIR, EMBEDDING_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the embedder model for Cyanview data")
    parser.add_argument("--docs_dir", type=str, required=True, help="Path to directory containing documents")
    parser.add_argument("--model_name", type=str, default=EMBEDDING_MODEL, 
                        help="Name of the base model to fine-tune")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save the fine-tuned model (default: data/fine_tuned_embedder)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    
    args = parser.parse_args()
    
    # Load documents for fine-tuning
    logger.info(f"Loading documents from {args.docs_dir}")
    documents = load_documents(args.docs_dir)
    
    # Extract training data
    training_data = []
    for doc in documents:
        # Extract data based on document type
        if doc.metadata.get('type') == 'qa':
            # For Q&A pairs
            content = doc.page_content
            q_parts = content.split("Q: ")
            if len(q_parts) > 1:
                question_part = q_parts[1]
                a_parts = question_part.split("A: ")
                if len(a_parts) > 1:
                    question = a_parts[0].strip()
                    answer = a_parts[1].strip()
                    
                    training_data.append({
                        'type': 'qa',
                        'question': question,
                        'answer': answer,
                        'content': doc.page_content
                    })
        else:
            # For regular documents
            if 'product' in doc.metadata:
                training_data.append({
                    'type': 'document',
                    'product': doc.metadata['product'],
                    'content': doc.page_content
                })
    
    logger.info(f"Extracted {len(training_data)} training items")
    
    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(DATA_DIR, "fine_tuned_embedder")
    
    # Fine-tune the model
    logger.info(f"Fine-tuning model {args.model_name} for {args.epochs} epochs")
    model = fine_tune_embedder(
        data=training_data,
        model_name=args.model_name,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()
