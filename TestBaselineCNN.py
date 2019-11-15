# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:45:07 2019

@author: Rohan M
"""
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.datasets import mnist
from tqdm import tqdm


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

json_data = []
accuracy = dict()
accuracy[6] = 0
accuracy[8] = 0
accuracy[10] = 0
accuracy[12] = 0
accuracy[14] = 0
accuracy[16] = 0
accuracy[18] = 0
accuracy[20] = 0
accuracy[22] = 0
accuracy[24] = 0
accuracy[26] = 0



for j in tqdm(range(X_test.shape[0])):
    for i in range(6,28,2):
        toPad = (32 - i)//2
        X_tester = np.pad(resize(X_test[j],(i,i)),[toPad,toPad],mode='constant')
        if np.argmax(model.predict(X_tester.reshape(1,32,32,1))) == y_test[j]:
            accuracy[i] +=1

print(accuracy)
for state in accuracy: 
    print("The Accuracy of image size:{} is:{}".format(state,accuracy[state]/X_test.shape[0])) 