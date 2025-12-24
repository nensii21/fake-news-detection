"""
Machine learning model for fake news detection.
"""
import pickle
import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class FakeNewsDetector:
    """
    Fake news detection model using TF-IDF and Logistic Regression.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False
    
    def train(self, X_train: List[str], y_train: List[int]) -> None:
        """
        Train the fake news detection model.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (0 = real, 1 = fake)
        """
        print("Vectorizing training data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        print("Training model...")
        self.model.fit(X_train_vectorized, y_train)
        self.is_trained = True
        print("Model training completed!")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict whether texts are fake or real.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            Array of predictions (0 = real, 1 = fake)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_vectorized = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vectorized)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            Array of probability scores [P(real), P(fake)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_vectorized = self.vectorizer.transform(texts)
        probabilities = self.model.predict_proba(X_vectorized)
        return probabilities
    
    def evaluate(self, X_test: List[str], y_test: List[int]) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: List of test text samples
            y_test: List of test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Real', 'Fake']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }
    
    def save(self, model_dir: str = "models") -> None:
        """
        Save the trained model and vectorizer.
        
        Args:
            model_dir: Directory to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {model_dir}/")
    
    def load(self, model_dir: str = "models") -> None:
        """
        Load a pre-trained model and vectorizer.
        
        Args:
            model_dir: Directory containing the saved model
        """
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        model_path = os.path.join(model_dir, 'model.pkl')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {model_dir}/")

