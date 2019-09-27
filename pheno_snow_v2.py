from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle
import random
import os

NAME = "phenocam-snow-cnn"

tensorBoard = TensorBoard(log_dir=f'logs/{NAME}')
train_data = "pictures/trainImages"
test_data = "pictures/testImages"


# Assign choice to array snow [1,0] or no snow [0,1]
def one_shot_label(img):
    label = img.split('.')[0]
    if label == 'snow':
        one_shot = np.array([1,0])
    else:
        one_shot = np.array([0,1])
    return one_shot

def train_data_with_label():
    train_images = []
    for index in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, index)
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128))
        train_images.append([np.array(img), one_shot_label(index)])
    random.shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for index in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, index)
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128))
        test_images.append([np.array(img), one_shot_label(index)])
    return test_images

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = keras.Sequential()

# 1st Convolution Layer
model.add(InputLayer(input_shape=[128,128,1]))
model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=5,padding='same'))

# 2nd Convolution Layer. More filters to reach more details.
# i.e Snow on ground not on trees, difference in glare and snow.
model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=5,padding='same'))

# 3rd Convolution Layer. More filters to reach more details.
# i.e Snow on ground not on trees, difference in glare and snow.
model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=5,padding='same'))

# Dropout Layer to avoid overfitting, Flatten before
# being fed into fully connected layers.
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(rate=0.5))
model.add(Activation("relu"))

# The number of neurons in the output layer will
# be equal to number of classes in the problem
model.add(Dense(2, activation= 'softmax'))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x=tr_img_data, y=tr_lbl_data,epochs=25)

# Plot predicted data of test images.
fig=plt.figure(figsize=(40,40))

for count, data in enumerate(testing_images[0:]):
    y = fig.add_subplot(6,5,count+1)
    img = data[0]
    data = img.reshape(1,128,128,-1)
    model_out = model.predict([data])

    # If the argmax of the prediction is 1 or [0,1]
    # then it has no snow
    if np.argmax(model_out) == 1:
        str_label = "No Snow"
    else:
        str_label = "Snow"

    y.imshow(img)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
