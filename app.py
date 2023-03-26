from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    file = request.files['file']
    
    # Read the file as an image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform keypoint detection and cropping here
    
    # Load the model
    model = keras.models.load_model('model.h5')
    
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape((1, 128, 128, 1))

    # Generate the cropped image using the model
    cropped_img = model.predict(img)
    cropped_img = cropped_img.reshape((128, 128))

    # Return the cropped image as a file
    _, img_encoded = cv2.imencode('.png', cropped_img)
    return img_encoded.tostring()


if __name__ == '__main__':
    app.run(debug=True)