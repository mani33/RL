# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:11:19 2024
Toy example neural network to learn to extract and set weights

@author: msubramaniyan
"""
# Import
import keras
from keras import Sequential
from keras.layers import Dense, Input, Activation
import numpy as np
#%%
x = np.array([[0,0],
              [1,0],
              [0,1],
              [1,1]])
y = np.array([0,1,1,0])
#%% Create model
keras.backend.clear_session()
model = Sequential(name='Three_layer_NN')
batch_size = 4
in_units = 2
model.add(Input(shape=(in_units,), batch_size=batch_size,name='Input layer'))
model.add(Dense(4, activation='relu', name='Hidden layer'))
model.add(Dense(1))
model.summary()
#%% Train
model.compile(optimizer='adam',loss='mean_squared_error')