# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:34:27 2019

@author: Rhythm G and Rohan M
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
import random
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


image_dims = (96, 96, 3)

print("[INFO] Loading Images..")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

# loop over the input images
for imagePath in tqdm(imagePaths):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (image_dims[0], image_dims[1]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
    
print(data[0])
print(labels[0])

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] Class Labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

print("[INFO] Shape of Vectors")
print(data.shape)
print(labels.shape)


# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

data_padded = data
labels_padded = labels
#resize images to 94 and Pad to 96 and print shape 
for image_size in tqdm(range(48,96,10),position=0, leave=True):
    X_temp, y = resizeImages(data, labels, resize_shape=(image_size,image_size))
    X_temp_padded_96,y = padToLength(X_temp, labels, padding_shape=(96,96))

    data_padded = np.concatenate((data_padded, X_temp_padded_96))
    labels_padded = np.concatenate((labels_padded, y))

print("[INFO] Shape of Vectors")
print(data_padded.shape)
print(labels_padded.shape)

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
model.add(Dense(len(mlb.classes_), activation="sigmoid"))

model.summary()
#Compile as usual
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training network...")

