"""
Text splitting utilities optimized for technical documentation.
"""
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks optimized for technical documentation.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of split document chunks
    """
    # Technical documentation-specific separators
    separators = [
        # Headers
        "\n## ",
        "\n### ",
        "\n#### ",
        # Lists
        "\n- ",
        "\n* ",
        "\n1. ",
        # Paragraphs
        "\n\n",
        "\n",
        # Sentences
        ". ",
        "? ",
        "! ",
        # Other separators
        ";",
        ":",
        " ",
        ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    # Process documents and preserve metadata
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc.page_content)
        
        for i, chunk_text in enumerate(doc_chunks):
            # Create new metadata dict with original metadata plus chunk info
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk"] = i
            chunk_metadata["chunk_total"] = len(doc_chunks)
            
            # For Q&A pairs, try to keep question and answer together
            if doc.metadata.get('type') == 'qa' and "Q: " in chunk_text and "A: " not in chunk_text:
                # Find the question and get the answer from original text
                question = chunk_text[chunk_text.find("Q: "):]
                full_text = doc.page_content
                answer_start = full_text.find("A: ", full_text.find(question))
                if answer_start != -1:
                    # Include the beginning of the answer if possible
                    answer_text = full_text[answer_start:answer_start+min(200, len(full_text)-answer_start)]
                    chunk_text = f"{question}\n{answer_text}..."
            
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    
    return chunks