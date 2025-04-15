"""
Fine-tuning utilities for embedder models.
"""
from typing import List, Dict, Optional
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_DIR, EMBEDDING_MODEL

def prepare_training_examples(data: List[Dict]) -> List[InputExample]:
    """
    Prepare training examples for fine-tuning from document data.
    
    Args:
        data: List of dictionaries with document data
        
    Returns:
        List of InputExample objects for training
    """
    train_examples = []
    
    for item in data:
        if item.get('type') == 'qa':
            # For Q&A pairs, create pairs of question and answer
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question and answer:
                # Create positive example (question-answer pair)
                train_examples.append(InputExample(texts=[question, answer]))
                
                # Also add query-content pair if available
                if 'content' in item:
                    train_examples.append(InputExample(texts=[question, item['content']]))
        
        elif 'content' in item and 'product' in item:
            # For product documentation, create pairs of product name and content
            product = item['product']
            content = item['content']
            
            if product and content:
                # Create product name and content pair
                train_examples.append(InputExample(texts=[f"About {product}", content]))
    
    return train_examples

def fine_tune_embedder(
    data: List[Dict], 
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 16
) -> SentenceTransformer:
    """
    Fine-tune a sentence transformer model on Cyanview-specific data.
    
    Args:
        data: List of dictionaries with document data
        model_name: Base model to fine-tune (defaults to config.EMBEDDING_MODEL)
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Fine-tuned SentenceTransformer model
    """
    # Set default values
    model_name = model_name or EMBEDDING_MODEL
    output_dir = output_dir or os.path.join(DATA_DIR, "fine_tuned_embedder")
    
    # Prepare training examples
    train_examples = prepare_training_examples(data)
    
    if not train_examples:
        raise ValueError("No training examples could be created from the provided data")
    
    print(f"Created {len(train_examples)} training examples")
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Prepare data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Initialize loss
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
    
    return model