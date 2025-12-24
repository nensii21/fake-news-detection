# Fake News Detection Project

A machine learning-based system for detecting fake news articles using Natural Language Processing (NLP) and text classification.

## Features

- **Machine Learning Model**: Uses TF-IDF vectorization and Logistic Regression for classification
- **Web Interface**: Beautiful, modern web UI for real-time news analysis
- **REST API**: JSON API endpoint for programmatic access
- **Easy to Use**: Simple training and prediction pipeline

## Project Structure

```
fake news detection/
├── src/
│   ├── preprocessing.py    # Data cleaning and preprocessing utilities
│   ├── model.py           # FakeNewsDetector class with ML model
│   └── train.py           # Training script
├── templates/
│   └── index.html         # Web interface
├── models/                # Saved models (created after training)
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

First, train the model using the training script:

```bash
python src/train.py
```

This will:
- Load and preprocess sample data
- Train the fake news detection model
- Evaluate the model performance
- Save the trained model to the `models/` directory

### 2. Run the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### 3. Use the API

You can also use the REST API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'
```

Response format:
```json
{
  "prediction": "Fake",
  "confidence": {
    "real": 0.15,
    "fake": 0.85
  },
  "text": "Your news article text here"
}
```

## Using Your Own Data

To train the model on your own dataset:

1. Create a CSV file with two columns:
   - `text`: The news article text
   - `label`: 0 for real news, 1 for fake news

2. Modify `src/preprocessing.py` to load your data:
   ```python
   texts, labels = load_and_preprocess_data('path/to/your/data.csv')
   ```

3. Run the training script again.

## Model Details

- **Vectorization**: TF-IDF with max 5000 features, English stop words, and bigrams
- **Classifier**: Logistic Regression
- **Features**: Text preprocessing includes:
  - Lowercasing
  - URL removal
  - Email removal
  - Special character removal
  - Whitespace normalization

## Example Predictions

The model can analyze various types of news articles:

- **Real News**: "Scientists have discovered a new planet that could support life"
- **Fake News**: "Breaking: Aliens have landed in New York City and are taking over"

## Health Check

Check if the API is running:
```bash
curl http://localhost:5000/health
```

## Requirements

- Python 3.7+
- Flask 3.0.0
- scikit-learn 1.3.2
- pandas 2.1.3
- numpy 1.26.2

## Future Improvements

- Add more sophisticated models (BERT, RoBERTa)
- Implement batch prediction
- Add model versioning
- Create a dataset collection tool
- Add confidence threshold tuning
- Implement model retraining pipeline

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

