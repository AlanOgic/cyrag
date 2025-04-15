"""
Main entry point for the Cyanview RAG system.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.query import rag_query
from src.utils.evaluation import evaluate_rag_system

def main():
    parser = argparse.ArgumentParser(description="Cyanview RAG System")
    parser.add_argument("--query", type=str, help="Query to run against the RAG system")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on the RAG system")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    if args.query:
        result = rag_query(args.query, top_k=args.top_k)
        print("\n=== Query ===")
        print(args.query)
        print("\n=== Retrieved Sources ===")
        for source in result["sources"]:
            print(f"- {source}")
        print("\n=== Answer ===")
        print(result["answer"] if "answer" in result else "No answer generated")
    
    elif args.evaluate:
        evaluate_rag_system()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()