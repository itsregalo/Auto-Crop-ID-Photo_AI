# %%
# Automatic generation system of student ID photos based on deep learning

# %%
"""
A model to be used in a webpage that can automatically complete the cropping of the uploaded photos according to the detected key point
information, and generate a standard version of the ID photo
"""

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c facial-keypoints-detection

!unzip facial-keypoints-detection.zip

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import glob as gb
import tensorflow as tf
import splitfolders
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping                                               


# %%
"""
file structure:
    - facial-keypoints-detection.zip
    - IdLookupTable.csv
    - SampleSubmission.csv
    - test.zip
    - training.zip
"""

with zipfile.ZipFile('training.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

with zipfile.ZipFile('test.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Load the training dataset into memory
train_data = pd.read_csv('data/training.csv')

train_data.head()

# %%
# test data
test_data = pd.read_csv('data/test.csv')
test_data.head()

# %%
# check the number of missing values
train_data.isnull().sum()

# using ffill to fill the missing values
train_data.fillna(method='ffill', inplace=True)

## Analysis of the Images
# %%
im_width, im_height = 96, 96

# %%
# checking the data type of the image column
type(train_data['Image'][0]) # str

# %%
# convert the image column to numpy array
train_data['Image'] = train_data['Image'].apply(lambda x: np.fromstring(x, sep=' '))

# %%
img = []

for i in range(0, 7049):
    img_pixel = train_data['Image'][i].reshape(im_width, im_height)
    img.append(img_pixel)

# %%
# convert the list to numpy array
img = np.array(img)

# %%
# pprint some images
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(img[0], cmap='gray')
ax[0, 1].imshow(img[1], cmap='gray')
ax[1, 0].imshow(img[2], cmap='gray')
ax[1, 1].imshow(img[3], cmap='gray')

# %%
# separate the features and labels
X = img
y = train_data.drop(['Image'], axis=1)

# %%
from sklearn.model_selection import train_test_split
# split the data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# convert the data to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# %%
# normalize the data
X_train /= 255
X_val /= 255

# %%
# reshape the data
X_train = X_train.reshape(-1, 96, 96, 1)
X_val = X_val.reshape(-1, 96, 96, 1)

# %%

# check the shape of the data
X_train.shape, X_val.shape, y_train.shape, y_val.shape

plt.imshow(X_train[1500])
plt.show()

# face keypoints
def show_keypoints(image, keypoints):
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[0::2], keypoints[1::2], marker='o', s=20, c='r')

# %%
plt.figure(figsize=(20, 10))
for i in range(0, 5):
    plt.subplot(1, 5, i+1)
    show_keypoints(X_train[i], y_train.iloc[i])

# %%
# define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(30)
])

# %%
# compile the model
model.compile(optimizer='adam',
                loss='mean_squared_error',          
                metrics=['mae'])

# %%
# define the callbacks
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

# %%
# train the model
history = model.fit(X_train, y_train, epochs=100, callbacks=[checkpoint, earlystop], validation_data=(X_val, y_val))

# %%
# plot the loss and accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# save the model
model.save('model.h5')

# %%
# load the model
model = keras.models.load_model('model.h5')

"""
A flask  webpage that can automatically complete the cropping of the uploaded photos according to the detected key point
information, and generate a standard version of the ID photo using the model trained above.
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# allowed extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # check if the request method is post
    if request.method == 'POST':
        

