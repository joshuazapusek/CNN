'''
Title: Convolutional Neural Network for Classifying Images 
Purpose: Accuracy >= 95% on Fashion MNIST Clothing Keras Dataset 
Author: Joshua Zapusek
Last Edited 7/15/2020
'''

# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense 
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import fashion_mnist

# PART 1: LOAD THE DATASET 

# Load the dataset 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

CLASSES = 10

# Print dimensions of train & test sets 
print('Train: X = %s, y = %s' % (x_train.shape, y_train.shape))
print('Test: X = %s, y = %s' % (x_test.shape, y_test.shape))

# Plot images with matplotlib
# NOTICE -- FIGURE OUT HOW TO USE MATPLOTLIB FOR THIS -- DONT KNOW HOW THIS CODE WORKS 
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap = plt.get_cmap('gray'))
plt.show()

# Convert to range (-1, +1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convet the Y-axis to categorical values 
y_train = to_categorical(y_train, CLASSES)
y_test = to_categorical(y_test, CLASSES)

# PART 2: BUILD THE MODEL USING FUNCTIONAL API

input_layer = Input(shape = (32, 32, 3))         # Each Image is size 32x32 pixels with RGB channel
x = Flatten()(input_layer)                       # Create vector of 32x32x3 magnitude 
x = Dense(units = 200, activiation = 'relu')(x)
x = Dense()
