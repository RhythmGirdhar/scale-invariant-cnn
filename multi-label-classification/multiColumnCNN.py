# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:35:55 2019

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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
import keras.losses



#random.shuffle(imagePaths)
image_dims = (96, 96, 3)
data_padded = np.load("data.npy")
labels_padded = np.load("labels.py")

(trainX, testX, trainY, testY) = train_test_split(data_padded,	labels_padded, test_size=0.2, random_state=42)

plotMnistImages(trainX, trainY)
# define two sets of inputs
inputA = Input(shape=(image_dims[0],image_dims[1],image_dims[2]))
 
#First Branch
cnn_1 = Conv2D(8, kernel_size=(3,3), padding="valid")(inputA)
cnn_1 = LeakyReLU(alpha=0.1)(cnn_1)
cnn_1 = Dropout(0.2)(cnn_1)

cnn_1 = Conv2D(16, kernel_size=(3,3), padding="valid")(cnn_1)
cnn_1 = LeakyReLU(alpha=0.1)(cnn_1)
cnn_1 = Dropout(0.2)(cnn_1)


cnn_1 = Conv2D(32, kernel_size=(3,3), padding="valid")(cnn_1)
cnn_1 = LeakyReLU(alpha=0.1)(cnn_1)
cnn_1 = MaxPooling2D(pool_size=(2,2))(cnn_1)
cnn_1 = Flatten()(cnn_1)
cnn_1 = Model(inputs=inputA, outputs=cnn_1)


#Second Branch
cnn_2 = Conv2D(8, kernel_size=(5,5), padding="valid")(inputA)
cnn_2 = LeakyReLU(alpha=0.1)(cnn_2)
cnn_2 = Dropout(0.2)(cnn_2)

cnn_2 = Conv2D(16, kernel_size=(5,5), padding="valid")(cnn_2)
cnn_2 = LeakyReLU(alpha=0.1)(cnn_2)
cnn_2 = Dropout(0.2)(cnn_2)


cnn_2 = Conv2D(32, kernel_size=(5,5), padding="valid")(cnn_2)
cnn_2 = LeakyReLU(alpha=0.1)(cnn_2)
cnn_2 = MaxPooling2D(pool_size=(2,2))(cnn_2)
cnn_2 = Flatten()(cnn_2)
cnn_2 = Model(inputs=inputA, outputs=cnn_2)

#Third Branch
cnn_3 = Conv2D(8, kernel_size=(7,7), padding="valid")(inputA)
cnn_3 = LeakyReLU(alpha=0.1)(cnn_3)
cnn_3 = Dropout(0.2)(cnn_3)

cnn_3 = Conv2D(16, kernel_size=(7,7), padding="valid")(cnn_3)
cnn_3 = LeakyReLU(alpha=0.1)(cnn_3)
cnn_3 = Dropout(0.2)(cnn_3)


cnn_3 = Conv2D(32, kernel_size=(7,7), padding="valid")(cnn_3)
cnn_3 = LeakyReLU(alpha=0.1)(cnn_3)
cnn_3 = MaxPooling2D(pool_size=(2,2))(cnn_3)
cnn_3 = Flatten()(cnn_3)
cnn_3 = Model(inputs=inputA, outputs=cnn_3)6

combined = concatenate([cnn_1.output, cnn_2.output, cnn_3.output])
z = Dense(128)(combined)
z = LeakyReLU(alpha=0.1)(z)
z = Dense(10, activation="sigmoid")(z)


model = Model(inputs=inputA, outputs=z)
model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Lets train

model.fit(trainX, trainY, epochs=3, batch_size=64, verbose=True)

#Validate Model
scores = model.evaluate(testX,testY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model_multicnn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_multicnn.h5")
print("Saved model to disk")