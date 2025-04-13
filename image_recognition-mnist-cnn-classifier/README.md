# README.md
# MNIST Digit Recognition Web Application

A modular web application for recognizing handwritten digits using the MNIST dataset and a CNN model built with TensorFlow/Keras.

## Features

- Draw or upload digits for recognition
- Interactive UI with canvas drawing
- Visual confidence display
- Alternative predictions
- Model training interface
- Example visualizations from the MNIST dataset

## Project Structure

```
mnist_app/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── models/                # Model definitions
│   └── mnist_model.py     # MNIST CNN model
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Application styles
│   └── js/
│   │   └── main.js        # Frontend JavaScript
├── templates/             # HTML templates
│   ├── index.html         # Main application page
│   ├── result.html        # Prediction results page
│   └── visualize.html     # Digit examples page
└── utils/                 # Utility functions
    └── image_processor.py # Image processing utilities
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to http://localhost:5000

## Usage

- **Draw Mode**: Use your mouse or finger to draw a digit on the canvas and click "Recognize"
- **Upload Mode**: Upload an image of a handwritten digit and click "Recognize Digit"
- **Visualize Examples**: Navigate to /visualize/{digit} to see examples from the MNIST dataset
- **Train Model**: Access /train?secret=admin-secret-key to retrain the model (admin only)

## Model Architecture

The CNN model consists of:
- Two convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Fully connected dense layers
- Softmax output layer for digit classification

## API Endpoints

- `GET /` - Main application page
- `POST /predict` - Submit an image file for prediction
- `POST /predict_canvas` - Submit canvas drawing for prediction
- `GET /train` - Train a new model (protected)
- `GET /visualize/<digit>` - View examples of a specific digit
- `GET /model_info` - Get information about the current model

## License

MIT