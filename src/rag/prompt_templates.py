"""
Prompt templates for the RAG system.
"""
from langchain.prompts import PromptTemplate

def get_rag_prompt() -> PromptTemplate:
    """
    Get the prompt template for RAG responses.
    
    Returns:
        PromptTemplate for RAG responses
    """
    template = """You are an expert on Cyanview camera control systems and technical documentation.
Based ONLY on the following context, answer the question.

Context:
{context}

Question: {query}

Instructions:
1. If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question."
2. Be specific and detailed, drawing from the technical information in the context.
3. When mentioning products, include their primary function (e.g., "RCP (Remote Control Panel)")
4. When answering about connections or integrations, mention specific cable types or protocols if available in the context.
5. Limit your response to information contained in the context. Do not include outside knowledge.

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "query"]
    )

def get_technical_rag_prompt() -> PromptTemplate:
    """
    Get a more technical prompt template for developer-focused responses.
    
    Returns:
        PromptTemplate for technical RAG responses
    """
    template = """You are a technical expert on Cyanview camera control systems with deep knowledge of their APIs, protocols, and integration specifications.
Based ONLY on the following technical documentation, answer the question with a developer/engineer focus.

Context:
{context}

Question: {query}

Instructions:
1. Focus on technical details like protocols, API calls, configuration parameters, and integration methods.
2. Include specific technical requirements such as cable specifications, port configurations, or firmware dependencies if mentioned in the context.
3. Use bullet points or code blocks where appropriate for clarity.
4. If specific version requirements are mentioned in the context, highlight them.
5. If the documentation is insufficient to answer the technical question completely, explicitly state what information is missing.

Technical Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "query"]
    )