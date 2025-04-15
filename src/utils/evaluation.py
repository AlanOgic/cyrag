"""
Evaluation utilities for the RAG system.
"""
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import statistics

from config import DATA_DIR
from src.rag.query import rag_query

def load_eval_queries() -> List[Dict[str, str]]:
    """
    Load evaluation queries and expected answers.
    
    Returns:
        List of dictionaries with queries and expected answers
    """
    eval_file = os.path.join(DATA_DIR, "eval_queries.json")
    
    # Check if eval file exists, if not create a sample one
    if not os.path.exists(eval_file):
        sample_queries = [
            {
                "query": "How does Cyanview RIO connect to cameras over internet?",
                "expected_answer": "RIO connects over the internet using its full license (CY-RIO) which supports WAN connectivity. It can use 4G/5G USB dongles and leverages Cyanview's Cloud Relay service, which facilitates remote connections without requiring open ports."
            },
            {
                "query": "What's the difference between RIO and RIO Live?",
                "expected_answer": "RIO Live is limited to LAN-only remote control, while the full RIO license (CY-RIO) supports WAN connectivity over the internet and cellular networks. RIO Live is designed for local live production, while full RIO enables remote production (REMI) over the internet."
            },
            {
                "query": "Which Cyanview cable is needed for B4 lens control?",
                "expected_answer": "For controlling B4 broadcast lenses, the CY-CBL-6P-B4-xx cable is required. This adapter cable connects to the lens's 12-pin Hirose connector to enable control via CI0 or RIO."
            }
        ]
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save sample queries
        with open(eval_file, "w") as f:
            json.dump(sample_queries, f, indent=2)
    
    # Load eval queries
    with open(eval_file, "r") as f:
        eval_queries = json.load(f)
    
    return eval_queries

def evaluate_query(query: str, expected_answer: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate a single query against the RAG system.
    
    Args:
        query: Query text
        expected_answer: Expected answer text
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with evaluation results
    """
    # Execute query
    start_time = time.time()
    result = rag_query(query, top_k=top_k)
    elapsed_time = time.time() - start_time
    
    # Calculate retrieval metrics
    retrieval_score = 0
    if "sources" in result and result["sources"]:
        retrieval_score = 1.0  # Basic retrieval success
    
    # If there's an answer, calculate simple content overlap
    answer_quality = 0
    if "answer" in result and result["answer"]:
        # Calculate token overlap between expected and actual answers
        expected_tokens = set(expected_answer.lower().split())
        actual_tokens = set(result["answer"].lower().split())
        
        if expected_tokens and actual_tokens:
            overlap = len(expected_tokens.intersection(actual_tokens))
            answer_quality = overlap / len(expected_tokens)
    
    # Return evaluation results
    return {
        "query": query,
        "expected_answer": expected_answer,
        "actual_answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieval_score": retrieval_score,
        "answer_quality": answer_quality,
        "elapsed_time": elapsed_time
    }

def evaluate_rag_system(eval_queries: Optional[List[Dict[str, str]]] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate the RAG system on a set of queries.
    
    Args:
        eval_queries: List of queries and expected answers
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with evaluation results
    """
    # Load eval queries if not provided
    if eval_queries is None:
        eval_queries = load_eval_queries()
    
    # Evaluate each query
    results = []
    for query_data in eval_queries:
        result = evaluate_query(
            query=query_data["query"],
            expected_answer=query_data["expected_answer"],
            top_k=top_k
        )
        results.append(result)
        
        # Print progress
        print(f"Query: {query_data['query']}")
        print(f"Retrieval Score: {result['retrieval_score']:.2f}")
        print(f"Answer Quality: {result['answer_quality']:.2f}")
        print(f"Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print("-" * 50)
    
    # Calculate aggregate metrics
    retrieval_scores = [r["retrieval_score"] for r in results]
    answer_qualities = [r["answer_quality"] for r in results]
    elapsed_times = [r["elapsed_time"] for r in results]
    
    avg_retrieval = statistics.mean(retrieval_scores) if retrieval_scores else 0
    avg_quality = statistics.mean(answer_qualities) if answer_qualities else 0
    avg_time = statistics.mean(elapsed_times) if elapsed_times else 0
    
    # Create evaluation summary
    summary = {
        "avg_retrieval_score": avg_retrieval,
        "avg_answer_quality": avg_quality,
        "avg_elapsed_time": avg_time,
        "num_queries": len(results),
        "results": results
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of Queries: {summary['num_queries']}")
    print(f"Average Retrieval Score: {summary['avg_retrieval_score']:.2f}")
    print(f"Average Answer Quality: {summary['avg_answer_quality']:.2f}")
    print(f"Average Elapsed Time: {summary['avg_elapsed_time']:.2f} seconds")
    
    # Save results
    results_file = os.path.join(DATA_DIR, f"eval_results_{int(time.time())}.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation results saved to {results_file}")
    
    return summary