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
    
    # Perform keypoint detection and crop face from image 
    face_cascade = cv2.CascadeClassifier(current_dir + '/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(current_dir + '/static/face.jpg', roi_color)

    # Load the model
    model = keras.models.load_model(current_dir + '/model.h5')

    # Resize the image to 48x48
    img = cv2.resize(roi_color, (48, 48))

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    img = img / 255.0

    # Reshape the image
    img = img.reshape(1, 48, 48, 1)

    # save the image
    cv2.imwrite(current_dir + '/static/face.jpg', roi_color)

    return render_template('index.html', prediction=model.predict_classes(img))



if __name__ == '__main__':
    app.run(debug=True)