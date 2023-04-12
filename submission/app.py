from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow import keras
import os
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    # Get the uploaded file
    file = request.files['file']

    # Load the image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    # Loop through the faces
    for (x, y, w, h) in faces:
        # Crop the face
        face = img[y:y+h, x:x+w]

        saved_image = cv2.imwrite(os.path.join(current_dir, 'saved_face.jpg'), face)

        # return the face as content
        with open(os.path.join(current_dir, 'saved_face.jpg'), 'rb') as f:
            face_content = f.read()

        # Encode the face content as base64
        encoded_face = base64.b64encode(face_content)

        context = {
            'cropped_image': encoded_face.decode('utf-8'),
        }

        return render_template('cropped.html', **context)


    # If no faces were detected, return an error message
    return 'No faces were detected in the uploaded image.'

if __name__ == '__main__':
    app.run(debug=True)
