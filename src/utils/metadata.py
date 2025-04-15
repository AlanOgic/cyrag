"""
Metadata extraction utilities for Cyanview documentation.
"""
import re
from typing import Dict, Any, List, Optional

# List of Cyanview products to identify in content
CYANVIEW_PRODUCTS = [
    "RCP", "Remote Control Panel",
    "CI0", "Camera Interface", 
    "RIO", "Remote I/O", 
    "RIO Live", "RIO-Live",
    "VP4", "Video Processor",
    "NIO", "Network I/O",
    "GWY", "External Gateway",
    "CY-RSBM", "CY-CI0BM", 
    "CY-TALLY-BOX"
]

# Key topics within Cyanview documentation
CYANVIEW_TOPICS = [
    "Camera Control", "Lens Control", "Tally", "REMI", "Remote Production",
    "Color Correction", "Shading", "Integration", "PTZ", "Gimbal",
    "IP Control", "Serial Control", "SDI Control", "USB Control",
    "Wireless Control", "API", "Firmware", "Configuration"
]

def extract_metadata_from_content(content: str) -> Dict[str, Any]:
    """
    Extract metadata from document content.
    
    Args:
        content: Document content text
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Extract product references
    products = extract_product_references(content)
    if products:
        metadata["product"] = products[0]  # Primary product
        metadata["all_products"] = products  # All mentioned products
    
    # Extract topics
    topics = extract_topics(content)
    if topics:
        metadata["topic"] = topics[0]  # Primary topic
        metadata["all_topics"] = topics  # All mentioned topics
    
    # Detect if content is a Q&A pair
    if is_qa_content(content):
        metadata["type"] = "qa"
    
    return metadata

def extract_product_references(content: str) -> List[str]:
    """
    Extract references to Cyanview products from content.
    
    Args:
        content: Document content text
        
    Returns:
        List of product references, most prominent first
    """
    found_products = []
    
    # Check for each product
    for product in CYANVIEW_PRODUCTS:
        # Count occurrences of the product name
        count = len(re.findall(r'\b' + re.escape(product) + r'\b', content))
        if count > 0:
            found_products.append((product, count))
    
    # Sort by number of occurrences, descending
    found_products.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the product names
    return [product for product, _ in found_products]

def extract_topics(content: str) -> List[str]:
    """
    Extract topic references from content.
    
    Args:
        content: Document content text
        
    Returns:
        List of topic references, most prominent first
    """
    found_topics = []
    
    # Check for each topic
    for topic in CYANVIEW_TOPICS:
        # Count occurrences of the topic
        count = len(re.findall(r'\b' + re.escape(topic) + r'\b', content))
        if count > 0:
            found_topics.append((topic, count))
    
    # Sort by number of occurrences, descending
    found_topics.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the topic names
    return [topic for topic, _ in found_topics]

def is_qa_content(content: str) -> bool:
    """
    Determine if content is in a question-answer format.
    
    Args:
        content: Document content text
        
    Returns:
        True if content appears to be a Q&A pair
    """
    # Check for Q: and A: pattern
    if re.search(r'Q:.*?A:', content, re.DOTALL):
        return True
    
    # Check for question marks followed by answers
    if re.search(r'\?\s+[A-Z]', content):
        return True
    
    return False