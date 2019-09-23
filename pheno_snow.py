from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
import os

NAME = "phenocam-snow-cnn"

tensorBoard = TensorBoard(log_dir=f'logs/{NAME}')
DATADIR = "pictures/"
CATEGORIES = ["noSnow","Snow"]

IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to snow or no snow
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

training_features = []
training_labels = []

for features, label in training_data:
    training_features.append(features)
    training_labels.append(label)

training_features = np.array(training_features).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open("training_features.pickle", "wb")
pickle.dump(training_features, pickle_out)
pickle_out.close()

pickle_out = open("training_labels.pickle", "wb")
pickle.dump(training_labels, pickle_out)
pickle_out.close()

training_features = pickle.load(open("training_features.pickle", "rb"))
training_labels = pickle.load(open("training_labels.pickle", "rb"))

# Scale values to range from 0 to 1
training_features = training_features / 255.0

model = keras.Sequential()
model.add(Conv2D(64, (3,3), input_shape = training_features.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(training_features, training_labels, epochs=10)

# A prediction is an array of 10 numbers. These describe the "confidence"
# of the model that the image corresponds to each of the
# 10 different articles of clothing.
predictions = model.predict(test_images)
print(predictions[0])

print(np.argmax(predictions[0]))

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
