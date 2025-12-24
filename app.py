"""
Flask web application for fake news detection.
"""
from flask import Flask, render_template, request, jsonify
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import FakeNewsDetector
from src.preprocessing import clean_text

app = Flask(__name__)

# Initialize detector
detector = FakeNewsDetector()

# Try to load pre-trained model, otherwise it will need to be trained
try:
    detector.load()
    print("Pre-trained model loaded successfully!")
except FileNotFoundError:
    print("Warning: No pre-trained model found. Please run train.py first.")
    print("The app will still work, but you'll need to train the model first.")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for fake news prediction.
    
    Expected JSON:
    {
        "text": "The news article text to analyze"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in the request body'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Check if model is trained
        if not detector.is_trained:
            return jsonify({
                'error': 'Model not trained. Please run train.py first.'
            }), 503
        
        # Clean and predict
        cleaned_text = clean_text(text)
        prediction = detector.predict([cleaned_text])[0]
        probabilities = detector.predict_proba([cleaned_text])[0]
        
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            },
            'text': text
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_trained': detector.is_trained
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

