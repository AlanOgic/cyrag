#!/usr/bin/env python
"""
Script to evaluate the Cyanview RAG system.
"""
import argparse
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import the local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.evaluation import evaluate_rag_system, load_eval_queries
from config import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Cyanview RAG system")
    parser.add_argument("--eval_file", type=str, help="Path to JSON file with evaluation queries")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load evaluation queries
    if args.eval_file:
        with open(args.eval_file, 'r') as f:
            eval_queries = json.load(f)
    else:
        eval_queries = load_eval_queries()
    
    # Run evaluation
    logger.info(f"Evaluating RAG system on {len(eval_queries)} queries")
    results = evaluate_rag_system(eval_queries, top_k=args.top_k)
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {args.output}")

if __name__ == "__main__":
    main()
