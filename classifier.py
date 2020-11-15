'''
Title: Convolutional Neural Network for Classifying Images 
Purpose: Accuracy >= 95% on Fashion MNIST Clothing Keras Dataset 
Author: Joshua Zapusek
Last Edited 7/15/2020
'''
# Imports 
import numpy as np
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.models import Modelks
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from sklearn.model_selection import KFold

# ///////////////////////////////////////////////////////////////////////////////////////////////////
# PART 1: Load and prep

# Load the dataset 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

CLASSES = 10

# Reshape for grey scale
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Convert to range (0, +1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convet the Y-axis to categorical values 
y_train = to_categorical(y_train, CLASSES)
y_test = to_categorical(y_test, CLASSES)

# //////////////////////////////////////////////////////////////////////////////////////////////////////
# PART 2: Build the model and train
scores = []
histories = []
kfold = KFold(n_folds, shuffle = True, random_state = 1)

for train_ix, test_ix in kfold.split(x_train):
    model = model_arch()
    trainX, trainY, testX, testY = x_train[train_ix], y_train[train_ix], x_train[test_ix], y_train[test_ix]
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    scores.append(acc)
    histories.append(history)

# //////////////////////////////////////////////////////////////////////////////////////////////////////
# PART 3: Evaluate
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)

# Functions
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
# Function to build architecture
def model_arch():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu', kernel_initializer= 'he_uniform'))
    model.add(Dense(10, activation = 'softmax'))
    # Hyperparameter: learning rate = 1%, momentum = 90%
    opt = SGD(lr = 0.01, momentum = 0.9)
    # compile with optimizing cross-entropy loss for accuracy
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(211)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(212)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()