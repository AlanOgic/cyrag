"""
Specialized loader for JSON files containing Q&A pairs.
"""
import json
from typing import List, Dict, Any, Optional
import os

from langchain.schema import Document

def load_json_qa(filepath: str) -> List[Document]:
    """
    Load Q&A pairs from JSON files with various structures.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        List of Document objects, each representing a Q&A pair.
    """
    documents = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filename = os.path.basename(filepath)
        
        # Handle list of Q&A objects
        if isinstance(data, list):
            for i, item in enumerate(data):
                if 'question' in item and ('answer' in item or 'response' in item):
                    content = f"Q: {item.get('question')}\nA: {item.get('answer', item.get('response', ''))}"
                    metadata = {
                        'source': filepath,
                        'type': 'qa',
                        'index': i,
                    }
                    
                    # Include metadata from the original item if available
                    if 'metadata' in item and isinstance(item['metadata'], dict):
                        for k, v in item['metadata'].items():
                            metadata[k] = v
                    
                    documents.append(Document(page_content=content, metadata=metadata))
        
        # Handle dictionary with 'faq' or 'cyanview_faq' key
        elif isinstance(data, dict):
            # Flatten nested FAQ structures
            if 'faq' in data:
                faq_data = data['faq']
                for i, item in enumerate(faq_data):
                    content = f"Q: {item.get('question')}\nA: {item.get('answer')}"
                    metadata = {
                        'source': filepath,
                        'type': 'qa',
                        'index': i,
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            # Handle nested FAQ categories
            elif 'cyanview_faq' in data:
                faq_categories = data['cyanview_faq']
                for category, items in faq_categories.items():
                    for i, item in enumerate(items):
                        content = f"Q: {item.get('question')}\nA: {item.get('answer')}"
                        metadata = {
                            'source': filepath,
                            'type': 'qa',
                            'category': category,
                            'index': i,
                        }
                        documents.append(Document(page_content=content, metadata=metadata))
    
    except Exception as e:
        print(f"Error loading JSON file {filepath}: {e}")
    
    return documents