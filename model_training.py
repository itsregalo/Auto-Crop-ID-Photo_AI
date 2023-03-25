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

# %%
# Define the model

# Define the model
def create_model():
    """
    Create a model to be used in the webpage
    """
    # Load the InceptionV3 model
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(local_weights_file)

    # Freeze the layers of the pre-trained model
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Define the last layer of the pre-trained model
    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    # Define the layers of the model
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(pre_trained_model.input, x)

    # Compile the model
    model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    return model


# Train the model
def train_model():
    """
    Train the model
    """
    # Define the training and validation data directories
    train_dir = 'data/train'
    validation_dir = 'data/validation'

    # Define the training and validation data generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Define the training and validation data generators
    train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(150, 150))
    validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=(150, 150))

    # Create the model
    model = create_model()

    # Define the model checkpoint and early stopping callbacks
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator, epochs=100, steps_per_epoch=100, validation_steps=50, verbose=2, callbacks=[checkpoint, early])

    return history


# Test the model
def test_model():
    """
    Test the model
    """
    # Load the model
    model = load_model('model.h5')

    # Define the test data directory
    test_dir = 'data/test'

    # Define the test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Define the test data generator
    test_generator = test_datagen.flow_from_directory(test_dir, batch_size=20, class_mode='binary', target_size=(150, 150))

    # Test the model
    loss, accuracy = model.evaluate(test_generator, steps=50, verbose=0)

    print('test loss:', loss)
    print('test accuracy:', accuracy)


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
    if i < 2000:
        os.rename(os.path.join(original_dir, file), os.path.join(train_dir, file))
    else:
        break

# move the next 500 files to the validation directory
for i, file in enumerate(os.listdir(original_dir)):
    if i < 500:
        os.rename(os.path.join(original_dir, file), os.path.join(validation_dir, file))
    else:
        break

# move the remaining files to the test directory
for i, file in enumerate(os.listdir(original_dir)):
    os.rename(os.path.join(original_dir, file), os.path.join(test_dir, file))
    


