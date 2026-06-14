import os
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf # Import TensorFlow to load Keras model from joblib

app = Flask(__name__)

# Load the trained model
# Ensure the model file 'handwritten_character_recognition_model.joblib' is in the same directory as app.py
model = joblib.load('handwritten_character_recognition_model.joblib')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction='No selected file')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Preprocess the image
            img = Image.open(filepath).convert('L') # Convert to grayscale
            img = img.resize((28, 28)) # Resize to 28x28 pixels
            img_array = np.array(img) # Convert to numpy array
            img_array = img_array.astype('float32') / 255.0 # Normalize
            img_array = img_array.reshape(1, 28, 28, 1) # Reshape for CNN input (batch, height, width, channels)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            return render_template('index.html', prediction=f'Predicted Digit: {predicted_class} (Confidence: {confidence:.2f}%)')
        except Exception as e:
            return render_template('index.html', prediction=f'Error processing image: {e}')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) # use_reloader=False for Colab/simple environments
