# Scale invariant CNN

"Size doesn't matter for me" - That's what our neural network said

## Keras - Baseline CNN
### Requirements
Install them first on your laptop
```
scikit-images
tqdm
```
### Baseline CNN
Took Mnist Data,  resized it to 16x16, 24x24 and 28x28. Then I padded all of them to make it look like 32x32

![alt text](https://gitlab.com/ssbhat98/scale-invariant-cnn/raw/master/images/plot_16_padded_32.png "plot_16_padded_32")
![alt text](https://gitlab.com/ssbhat98/scale-invariant-cnn/raw/master/images/plot_24_padded32.png "plot_24_padded32")
![alt text](https://gitlab.com/ssbhat98/scale-invariant-cnn/raw/master/images/plot_28_padded32.png "plot_28_padded32")

Training Loss and Accuracy:
```
loss: 0.0620 - acc: 98.23
```

Validation Accuracy:
```
acc:98.44444444444445
```

To Run it just do:
```
python3 BaselineCNN.py
```
