"""
Training script for fake news detection model.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import FakeNewsDetector
from src.preprocessing import load_and_preprocess_data, split_data


def main():
    """
    Main training function.
    """
    print("=" * 60)
    print("Fake News Detection Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    texts, labels = load_and_preprocess_data()
    print(f"   Loaded {len(texts)} samples")
    
    # Split data
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(texts, labels, test_size=0.2)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Initialize and train model
    print("\n3. Initializing model...")
    detector = FakeNewsDetector()
    
    print("\n4. Training model...")
    detector.train(X_train, y_train)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    detector.evaluate(X_test, y_test)
    
    # Save model
    print("\n6. Saving model...")
    detector.save()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

