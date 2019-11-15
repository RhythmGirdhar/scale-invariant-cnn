# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:46:02 2019

@author: Rohan M
"""

#Import Modules
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.datasets import mnist
from utils import resizeImages,padToLength, plotMnistImages

import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import keras.losses

#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_16, y = resizeImages(X_train, y_train, resize_shape=(16,16))
X_16_padded_32,y = padToLength(X_16, y_train, padding_shape=(32,32))
plotMnistImages(X_16_padded_32, y_train)

X_24, y = resizeImages(X_train, y_train, resize_shape=(24,24))
X_24_padded_32,y = padToLength(X_24, y_train, padding_shape=(32,32))
plotMnistImages(X_24_padded_32, y_train)

X_28_padded_32, y = padToLength(X_train, y_train, padding_shape=(32,32))
plotMnistImages(X_28_padded_32, y_train)
plt.show()

#Concatenate to get training data
train_x= np.concatenate((X_28_padded_32, X_24_padded_32, X_16_padded_32))
print(train_x.shape)
train_y = np.concatenate((y_train, y_train, y_train))
print(train_y.shape)

#Reshape & Normalize Data
train_x = train_x.reshape(-1, 32,32, 1)
train_x = train_x.astype('float32')
train_x = train_x / 255

#Convert the labels into one-hot
train_y_one_hot = keras.utils.to_categorical(train_y)

#Split data into training and validation
train_x,valid_x,train_label,valid_label = train_test_split(train_x, 
                                                           train_y_one_hot, 
                                                           test_size=0.2)
														   
														   
#We are ready. Build Model. 3 Convolutional layers, followed by fully connected
model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3), padding="valid", input_shape=(32,32,1)))
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
model.add(Dense(10, activation="softmax"))

model.summary()
#Compile as usual
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Lets train

model.fit(train_x, train_label, epochs=6, batch_size=64, verbose=True)

#Validate Model
model.evaluate(valid_x,valid_label)