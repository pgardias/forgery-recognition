import os
import sys
import glob
from PIL import Image
import numpy as np
from collections import defaultdict
import itertools

from os import listdir
from matplotlib import image
from PIL import Image

real_images = glob.glob('./blurred_data/real/*.png')
forged_images = glob.glob('./blurred_data/fake/*.png')


def reverse(s):
  str = ""
  for i in s:
    str = i + str
  return str

def get_image_id(image_path):
    """returns image ID from the image path"""
    image_id = ""
    found = False
    i = 3
    while not found:
        image_id += image_path[-1-i-1:-1-i]
        i+=1
        if image_path[-1-i-1:-1-i].isnumeric():
            found = True
    image_id = reverse(image_id)
    return image_id


real_images_dict = defaultdict(list)
forged_images_dict = defaultdict(list)

# Iterate over real images and put them in dictionary values for same image_id key.
for real_image, forged_image in zip(real_images, forged_images):
    # add image to dictionary
    real_image_id = get_image_id(real_image)
    real_images_dict[real_image_id].append(real_image)

    forged_image_id = get_image_id(forged_image)
    forged_images_dict[forged_image_id].append(forged_image)

negative_image_tuples = list()

for image_id in real_images_dict.keys():
    real = real_images_dict[image_id]
    forged = forged_images_dict[image_id]

    negative_image_tuples.extend(list(itertools.product(real, real, forged)))


def process(image_path):
    """returns processed images"""
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array_processed = 1 - image_array
    image_array_processed = image_array_processed / np.std(image_array_processed)
    image_array_processed = np.expand_dims(image_array_processed, axis=2)

    return image_array_processed


image_1 = []
image_2 = []
image_3 = []
labels = []

for anchor, positive, negative in negative_image_tuples[:1000]:
    image_1.append(process(anchor))
    image_2.append(process(positive))
    image_3.append(process(negative))
    labels.append(0)

image_1_array = np.asarray(image_1)
image_2_array = np.asarray(image_2)
image_3_array = np.asarray(image_3)
labels_array = np.array(labels)


idx = np.random.choice(range(len(image_1)), size=len(image_1), replace=False)

X_1 = image_1_array[idx]
X_2 = image_2_array[idx]
X_3 = image_3_array[idx]
y = labels_array[idx]

train_split = 0.8
valid_split = 0.9
train_offset = int(train_split * len(X_1))
valid_offset = int(valid_split * len(X_1))

X_1_train = X_1[:train_offset]
X_2_train = X_2[:train_offset]
X_3_train = X_3[:train_offset]
y_train = y[:train_offset]

X_1_valid = X_1[train_offset:valid_offset]
X_2_valid = X_2[train_offset:valid_offset]
X_3_valid = X_3[train_offset:valid_offset]
y_valid = y[train_offset:valid_offset]

X_1_test = X_1[valid_offset:]
X_2_test = X_2[valid_offset:]
X_3_test = X_3[valid_offset:]
y_test = y[valid_offset:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Dropout, Activation, Input, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from keras.layers.merge import Concatenate
from tensorflow.keras.layers import Lambda
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from tensorflow.keras.regularizers import l2
from keras import backend as K

def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net

model = get_siamese_model((28, 140, 1))
model.summary()

def identity_loss(y_true, y_pred):
    """
    Fake loss function for Keras. real loss function is L1 distance in model
    """
    return y_pred

model.compile(loss=identity_loss, optimizer=Adam(lr = 0.00006),metrics=["accuracy"])

nepochs=20
model.fit([X_1_train, X_3_train], y_train,
          batch_size=64,
          epochs=nepochs,
          validation_data=([X_1_valid, X_3_valid], y_valid))

print(model.predict([X_1_test, X_2_test]))
print(model.predict([X_1_test, X_3_test]))



