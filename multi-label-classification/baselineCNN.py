# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:34:27 2019

@author: Rhythm G and Rohan M
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import resizeImages, padToLength, plotMnistImages
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import keras.losses

#random.shuffle(imagePaths)
image_dims = (96, 96, 3)
data_padded = np.load("data.npy")
labels_padded = np.load("labels.npy")

(trainX, testX, trainY, testY) = train_test_split(data_padded,	labels_padded, test_size=0.2, random_state=42)

print("[INFO] compiling model...")
model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3), padding="valid", input_shape=(image_dims[0],image_dims[1],image_dims[2])))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Conv2D(16, kernel_size=(3,3), padding="valid"))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))


model.add(Conv2D(32, kernel_size=(3,3), padding="valid"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(6, activation="sigmoid"))

model.summary()
#Compile as usual
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training network...")
model.fit(trainX, trainY, epochs=2, batch_size=64, verbose=True)
print("[INFO] evaluate network...")

print(model.evaluate(testX,testY))

