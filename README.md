# Cyanview RAG System

A Retrieval-Augmented Generation (RAG) system specifically designed for Cyanview technical documentation, using Qdrant as a vector database and a specialized embedding model.

## Features

- Document loading for PDF, DOCX, TXT, and JSON files
- Specialized chunking for technical documentation
- Technical domain-optimized embeddings
- Qdrant vector storage with metadata filtering
- Query interface with context-aware response generation
- Evaluation tools to measure system performance
- Optional embedding fine-tuning

## Getting Started

### Prerequisites

- Python 3.8+
- Qdrant server (local or cloud)
- Access to embedding models

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your configuration values

### Usage

1. Ingest documents:
   ```
   python scripts/ingest.py --docs_dir path/to/cyanview_docs
   ```

2. Run the API:
   ```
   python -m api.app
   ```

3. Query the system:
   ```
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query":"How does Cyanview RIO connect to cameras over internet?"}'
   ```

## Project Structure

- `src/`: Core functionality
  - `data/`: Document loading and processing
  - `embeddings/`: Embedding models and utilities
  - `qdrant/`: Qdrant client and collection management
  - `rag/`: RAG query processing
- `api/`: FastAPI web service
- `scripts/`: Utility scripts for ingestion and evaluation

## License

This project is proprietary and confidential.