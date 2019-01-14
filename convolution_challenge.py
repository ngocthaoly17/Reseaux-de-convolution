# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:45:29 2018

@author: ngocthaoly
"""

from scipy.ndimage import imread
import os
import numpy as np
'''train walk'''
os.chdir("C:/Users/charl/OneDrive/Documents/jedha/full_time_exo/S6/challenge/train/walk")
listfiles = os.listdir()
walk_train = [imread(a) for a in listfiles]

walk_train_BW_2D = [np.mean(a[:,:,:-1],axis = 2) for a in walk_train]
walk_train_BW_2D_array = np.array(walk_train_BW_2D)
shape = walk_train_BW_2D_array.shape
walk_train_BW_2D_array = np.reshape(walk_train_BW_2D_array,(shape[0],1,shape[1],shape[2]))

y_train_walk = np.zeros(shape = walk_train_BW_2D_array.shape[0])

'''train run'''
os.chdir("C:/Users/charl/OneDrive/Documents/jedha/full_time_exo/S6/challenge/train/run")
listfiles = os.listdir()
run_train = [imread(a) for a in listfiles]

run_train_BW_2D = [np.mean(a[:,:,:-1],axis = 2) for a in run_train]
run_train_BW_2D_array = np.array(run_train_BW_2D)
shape = run_train_BW_2D_array.shape
run_train_BW_2D_array = np.reshape(run_train_BW_2D_array,(shape[0],1,shape[1],shape[2]))

y_train_run = np.zeros(shape = run_train_BW_2D_array.shape[0]) + 1

'''test walk'''
os.chdir("C:/Users/charl/OneDrive/Documents/jedha/full_time_exo/S6/challenge/test/walk")
listfiles = os.listdir()
walk_test = [imread(a) for a in listfiles]

walk_test_BW_2D = [np.mean(a[:,:,:-1],axis = 2) for a in walk_test]
walk_test_BW_2D_array = np.array(walk_test_BW_2D)
shape = walk_test_BW_2D_array.shape
walk_test_BW_2D_array = np.reshape(walk_test_BW_2D_array,(shape[0],1,shape[1],shape[2]))

y_test_walk = np.zeros(shape = walk_test_BW_2D_array.shape[0])

'''test run'''
os.chdir("C:/Users/charl/OneDrive/Documents/jedha/full_time_exo/S6/challenge/test/run")
listfiles = os.listdir()
run_test = [imread(a) for a in listfiles]

run_test_BW_2D = [np.mean(a[:,:,:-1],axis = 2) for a in run_test]
run_test_BW_2D_array = np.array(run_test_BW_2D)
shape = run_test_BW_2D_array.shape
run_test_BW_2D_array = np.reshape(run_test_BW_2D_array,(shape[0],1,shape[1],shape[2]))

y_test_run = np.zeros(shape = run_test_BW_2D_array.shape[0]) + 1

''' separate x et y '''
X_train = np.concatenate((walk_train_BW_2D_array,run_train_BW_2D_array))
X_test = np.concatenate((walk_test_BW_2D_array,run_test_BW_2D_array))
y_train = np.concatenate((y_train_walk,y_train_run))
y_test = np.concatenate((y_test_walk,y_test_run))
import keras as k
y_train = k.utils.to_categorical(y_train)
y_test = k.utils.to_categorical(y_test)

'''OBJECTIF MEILLEUR SCORE POSSIBLE DE PREDICTION RUN WALK SUR LES DONNEES DE TEST'''
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 224, 224), activation='tanh'))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=None, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
