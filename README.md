%%writefile README.md
# Handwritten Character Recognition Project (CodeAlpha ML Internship)

This repository contains a solution for a Handwritten Character Recognition project using a Convolutional Neural Network (CNN) trained on the MNIST dataset. It includes the Colab notebook used for model development, a trained model file, and a Flask web application to demonstrate the model's predictions.

## Project Components

1.  **`Handwritten_Character_Recognition_Colab.ipynb`**: The Google Colab notebook detailing the entire machine learning pipeline:
    *   Loading and preprocessing the MNIST dataset.
    *   Building and training a CNN using TensorFlow/Keras.
    *   Evaluating model performance (accuracy, loss, confusion matrix).
    *   Visualizing training history and sample predictions.
    *   Saving the trained model.

2.  **`handwritten_character_recognition_model.joblib`**: The trained CNN model saved in Joblib format.

3.  **`app.py`**: A Flask web application to serve the trained model. Users can upload an image of a handwritten digit, and the app will return the model's prediction.

4.  **`templates/index.html`**: The HTML template for the Flask application's user interface.

5.  **`requirements.txt`**: A list of all Python dependencies required to run the Flask application and reproduce the model training.

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/handwritten-character-recognition.git
    cd handwritten-character-recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure the model file is present:**
    Make sure `handwritten_character_recognition_model.joblib` is in the root directory alongside `app.py`. If you trained the model in the Colab notebook, download this file from there.

## Running the Flask Application

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000/`. You can now upload an image of a handwritten digit and get a prediction.

## Model Details

*   **Architecture**: Convolutional Neural Network (CNN).
*   **Dataset**: MNIST (60,000 training images, 10,000 testing images of handwritten digits).
*   **Preprocessing**: Pixel normalization (0-255 to 0-1), image reshaping (28x28 to 28x28x1).
*   **Training**: 5 epochs.
*   **Evaluation (on test set)**:
    *   Accuracy: ~99.23%
    *   Loss: ~0.0266

## Usage

*   Upload a grayscale image (PNG, JPG) of a single handwritten digit (0-9) where the digit is clearly visible.
*   The image will be resized to 28x28 pixels and processed by the CNN.
*   The application will display the predicted digit and the model's confidence in that prediction.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if you choose to include one).

## Acknowledgements

*   CodeAlpha for the Machine Learning Internship opportunity.
*   TensorFlow and Keras for the deep learning framework.
*   MNIST dataset for handwritten digit recognition research.
