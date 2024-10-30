# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:30:22 2024
Play with permuation invariance layer using demo code from:
https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/text_sum.ipynb
@author: msubramaniyan
"""
#%% Import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from IPython.display import SVG
import keras.backend as ker_bkend
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# from keras.utils.vis_utils import model_to_dot
# from tqdm import tqdm,trange!N14Je
#%%

#%%
def gen_data(num_train_examples, max_train_length):
    X = np.zeros((num_train_examples,max_train_length))
    sum_X = np.zeros((num_train_examples))
    # for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
    for i in range(num_train_examples):
        n = np.random.randint(1,max_train_length)
        for j in range(1,n+1):
            X[i,-j] = np.random.randint(1,10)
        sum_X[i] = np.sum(X[i])
    return X, sum_X

def get_deepset_model(max_length):
    input_txt = Input(shape=(max_length,)) # shape: not including batch size
    x = Embedding(11, 100, mask_zero=True)(input_txt)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: tf.keras.backend.sum(x, axis=1), 
                   output_shape=(lambda shape: (shape[0], shape[2])))
    # added = keras.layers.Add(x)    
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_txt, encoded)
    adam = Adam(learning_rate=1e-4, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer

#%% Train the model
# model
num_train_examples = 100000
max_train_length = 10

num_test_examples = 10000
min_test_length=5
max_test_length=100
step_test_length=5

model = get_deepset_model(max_train_length)
checkpointer = ModelCheckpoint(filepath='/weights/weights.keras', verbose=0, save_best_only=True)

X, sum_X = gen_data(num_train_examples, max_train_length)
#%%
model.fit(X, sum_X, epochs=10, batch_size=128,
        shuffle=True, validation_split=0.0123456789,
        callbacks=[checkpointer])

# model = load_model('/weights/weights.h5')

# # save weights
# deep_we = []
# for i in [1,2,4]:
#     w = model.get_layer(index=i).get_weights()
#     deep_we.append(w)




