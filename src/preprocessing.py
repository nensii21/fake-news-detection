"""
Data preprocessing utilities for fake news detection.
"""
import re
import string
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_preprocess_data(file_path: str = None) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess data from CSV file or create sample data.
    
    Args:
        file_path: Path to CSV file with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels) where labels are 0 (real) or 1 (fake)
    """
    if file_path:
        try:
            df = pd.read_csv(file_path)
            texts = df['text'].apply(clean_text).tolist()
            labels = df['label'].tolist()
            return texts, labels
        except Exception as e:
            print(f"Error loading file: {e}. Using sample data instead.")
    
    # Sample data for demonstration
    sample_texts = [
        "Scientists have discovered a new planet that could support life",
        "Breaking: Aliens have landed in New York City and are taking over",
        "New study shows that exercise improves mental health significantly",
        "Government officials confirm that vaccines contain microchips for tracking",
        "Climate change is causing severe weather patterns worldwide",
        "The moon landing was completely faked by NASA in a Hollywood studio",
        "Technology companies are developing AI to help solve global problems",
        "Celebrities are secretly controlling the world through mind control",
        "Medical research has made significant advances in cancer treatment",
        "The earth is flat and all space agencies are lying to us"
    ]
    
    sample_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = real, 1 = fake
    
    texts = [clean_text(text) for text in sample_texts]
    return texts, sample_labels


def split_data(texts: List[str], labels: List[int], test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.
    
    Args:
        texts: List of text strings
        labels: List of labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)

