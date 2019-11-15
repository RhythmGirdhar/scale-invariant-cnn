# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:02:38 2019

@author: Rohan M
"""
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

def plotMnistImages(X,y):
  fig = plt.figure()
  for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X[i], cmap='gray', interpolation='none')
    plt.title('Digit{}'.format(y[i]))
    plt.xticks([])
    plt.yticks([])


def resizeImages(X,y,resize_shape = (32,32)):
  X_resized = np.array([resize(i, resize_shape) for i in tqdm(X)])
  return X_resized, y

def padToLength(X,y,padding_shape = (32,32)):
  toPad = padding_shape[0] - X.shape[1]
  X_padded = np.array([np.pad(i,[toPad//2, toPad//2], mode='constant') for i in tqdm(X)])
  return X_padded,y