from typing import Union
import string
import re

def normalize_text(s: str) -> str:
    """
    Normalize text by removing articles, punctuation, and standardizing whitespace.
    
    Args:
        s (str): Input text to normalize
        
    Returns:
        str: Normalized text with standardized format
    """
    def remove_articles(text: str) -> str:
        """
        Remove articles 'a', 'an', 'the' from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with articles removed
        """
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        """
        Standardize whitespace in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with standardized whitespace
        """
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with punctuation removed
        """
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lowercase text
        """
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction: str, truth: str) -> int:
    """
    Compute exact match score between prediction and truth.
    
    Args:
        prediction (str): Predicted answer text
        truth (str): Ground truth answer text
        
    Returns:
        int: 1 if exact match after normalization, 0 otherwise
    """
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction: str, truth: str) -> float:
    """
    Compute F1 score between prediction and truth.
    
    Args:
        prediction (str): Predicted answer text
        truth (str): Ground truth answer text
        
    Returns:
        float: F1 score between 0 and 1, where 1 indicates perfect overlap
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # Handle empty answers
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # Handle no overlap case
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall) 