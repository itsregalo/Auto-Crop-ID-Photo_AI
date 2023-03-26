# %%
# Automatic generation system of student ID photos based on deep learning
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ashwingupta3012/human-faces

!unzip human-faces.zip

# %%
"""
A model to be used in a webpage that can automatically complete the cropping of the uploaded photos according to the detected key point
information, and generate a standard version of the ID photo
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# rename all the files in the directory as .jpg to facilitate the use of the model
original_dir = 'Humans'
new_dir = 'data'

# Create the new directory
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# rename all the files in the directory
for i, file in enumerate(os.listdir(original_dir)):
    if file.endswith('.jpg'):
        os.rename(os.path.join(original_dir, file), os.path.join(original_dir, str(i) + '.jpg'))

    if file.endswith('.jpeg'):
        os.rename(os.path.join(original_dir, file), os.path.join(original_dir, str(i) + '.jpg'))

    if file.endswith('.png'):
        os.rename(os.path.join(original_dir, file), os.path.join(original_dir, str(i) + '.png'))

    if file.endswith('.JPG'):
        os.rename(os.path.join(original_dir, file), os.path.join(original_dir, str(i) + '.jpg'))

# check the number of files in the directory
print(len(os.listdir(original_dir))) # 3817

# %%
# Split the data into training, validation, and test sets

# Create the training, validation, and test directories
train_dir = os.path.join(new_dir, 'train')
validation_dir = os.path.join(new_dir, 'validation')
test_dir = os.path.join(new_dir, 'test')

# Create the training, validation, and test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

"""
We want to use the data to train the model to detect faces so that we can crop the photos according to the detected key point information.
Therefore, we need to split the data into training, validation, and test sets, and then split the training set into training and validation sets.
"""

# move the first 2000 files to the training directory
for i, file in enumerate(os.listdir(original_dir)):
    if i < 5000:
        os.rename(os.path.join(original_dir, file), os.path.join(train_dir, file))
    else:
        break

# move the next 500 files to the validation directory
for i, file in enumerate(os.listdir(original_dir)):
    if i < 1000:
        os.rename(os.path.join(original_dir, file), os.path.join(validation_dir, file))
    else:
        break

# move the remaining files to the test directory
for i, file in enumerate(os.listdir(original_dir)):
    os.rename(os.path.join(original_dir, file), os.path.join(test_dir, file))
    

# check the number of files in the training, validation, and test directories
print(len(os.listdir(train_dir))) # 5000
print(len(os.listdir(validation_dir))) # 1000
print(len(os.listdir(test_dir))) # 1219

# %%
# model training

# Create the base model from the pre-trained model InceptionV3
base_model = InceptionV3(input_shape = (150, 150, 3), # Shape of our images
                            include_top = False, # Leave out the last fully connected layer     
                            weights = 'imagenet')

# Freeze the base model
base_model.trainable = False

# Add a classification head
maxpool_layer = layers.GlobalMaxPooling2D()
prediction_layer = layers.Dense(1, activation='sigmoid')
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

# Compile the model
model.compile(optimizer = RMSprop(lr=0.0001),
                loss = 'binary_crossentropy',           
                metrics = ['accuracy'])

# %%
# Data preprocessing

# Define the training and validation data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define the training and validation data generators
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='raw', target_size=(150, 150))
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='raw', target_size=(150, 150))

"""
why is it getting 0 images?: 

