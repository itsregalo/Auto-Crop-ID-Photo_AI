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
    
    # Read the file as an image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform keypoint detection and cropping here
    
    # Load the model
    model = keras.models.load_model('model.h5')
    
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = img.reshape((1, 96, 96, 1))

    # Generate the cropped image using the model
    cropped_img = model.predict(img)
    cropped_img = cv2.resize(cropped_img[0], (96, 96))
    interpolation = cv2.INTER_NEAREST if cropped_img.shape[-1] == 1 else cv2.INTER_CUBIC
    cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]), interpolation=interpolation)

    # Save the cropped image
    image_path = os.path.join(current_dir, 'cropped_img.jpg')
    cv2.imwrite(image_path, cropped_img)

    print('Image saved successfully')

    return render_template('index.html', cropped_img='cropped_img.jpg')

if __name__ == '__main__':
    app.run(debug=True)