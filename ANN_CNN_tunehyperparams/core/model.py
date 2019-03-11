# Written by Adrianna - 2019

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD
from keras.constraints import maxnorm

def create_model(filts=64, k_size=2, pool_size=2, neurons=10, optimizer='adam', activation='relu', dropout_rate=0.0):
    """
    Function to create model, required for KerasRegressor
    """

    # Define model
    model = Sequential()
    model.add(Conv1D(filters=filts, kernel_size=k_size, activation=activation, input_shape=(29, 3)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=optimizer, loss='mse')
    return model 
