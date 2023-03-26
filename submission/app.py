from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow import keras
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    file = request.files['file']
    
    # Read the image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Check the image dimensions
    if img.shape[0] < 96 or img.shape[1] < 96:
        raise ValueError('Image is too small!')

    # Resize the image to match the expected input shape of the model
    img = cv2.resize(img, (96, 96))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the model
    model = keras.models.load_model(os.path.join(current_dir, 'model.h5'))

    # Predict the image
    pred = model.predict(gray.reshape(-1, 96, 96, 1))

    # Get the predicted image
    pred = pred.reshape(96, 96)

    # Convert to uint8
    pred = pred.astype(np.uint8)

    # Save the image
    cv2.imwrite(os.path.join(current_dir, 'static/predicted.png'), pred)

    return render_template('index.html', predicted='predicted.png')


if __name__ == '__main__':
    app.run(debug=True)

    