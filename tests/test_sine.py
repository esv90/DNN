# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:24:04 2020

@author: a339594
"""

import numpy as np
import matplotlib.pyplot as plt

from ann.layers import Input, Dense
from ann.models import Sequential

from ann.optimizers import SGD

# Attempt to fit a sine wave with a neuron net

features = np.arange(0, 4*np.pi, 0.01)
features = features.reshape((features.shape[0], 1, 1))
labels = np.sin(features) + 1

# Shuffle data
idx = np.arange(len(features))
np.random.shuffle(idx)
features_shuffled = features[idx]
labels_shuffled = labels[idx]


input1 = Input(shape=(1,1), name='input1')
hidden1 = Dense(32, activation='tanh', name='hidden1')
hidden2 = Dense(32, activation='tanh', name='hidden2')
output1 = Dense(1, activation='linear', name='output1')

model1 = Sequential()
model1.add(input1, hidden1, hidden2, output1)
model1.compile(SGD(learning_rate=0.01), loss='mse', metrics=['mse'])


epochs = 10
# def train_model():
history = model1.fit(features_shuffled, labels_shuffled, verbose=2, batch_size=32, epochs=epochs)

plt.figure()
plt.plot(np.arange(epochs), history)
plt.gca().set_yscale('log')
plt.grid('on')
plt.gca().set_ylim([1e-4, 1e-0])
# return history

# history = train_model()

x = features
y = model1.predict(x, labels.shape)
plt.figure()
plt.plot(x[:, 0, 0], y[:, 0, 0], features[:, 0, 0], labels[:, 0, 0])
plt.show()