# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:24:04 2020

@author: a339594
"""

import numpy as np
import matplotlib.pyplot as plt

from ann.layers import Input, Dense
from ann.models import Sequential

from ann.optimizers import GradientDescent

# Matt Mazur's example
# features = np.array([np.array([[0.05], [0.10]]),
#                      np.array([[0.05], [0.10]]),
#                      np.array([[0.05], [0.10]]),
#                      np.array([[0.05], [0.10]])])
# labels = np.array([np.array([[0.01], [0.99]]),
#                    np.array([[0.01], [0.99]]),
#                    np.array([[0.01], [0.99]]),
#                    np.array([[0.01], [0.99]])])
#


features = np.array([[0.05, 0.10],
                     [0.05, 0.10],
                     [0.05, 0.10],
                     [0.05, 0.10]])

features = features.reshape(features.shape[0],features.shape[1], 1)

labels = np.array([[0.01, 0.99],
                   [0.01, 0.99],
                   [0.01, 0.99],
                   [0.01, 0.99]])

labels = labels.reshape(labels.shape[0], labels.shape[1], 1)

input1 = Input(input_shape=(2,1))
# input1.forward_pass(features[0])

hidden1 = Dense(2, activation='sigmoid')
# hidden1.set_input_layer(input1)
# hidden1.set_weights(np.array([[0.15, 0.20],
#                               [0.25, 0.30]]))
# hidden1.set_bias(np.array([[0.35],
#                            [0.35]]))

output1 = Dense(2, activation='sigmoid')
# output1.set_input_layer(hidden1)
# output1.set_weights(np.array([[0.40, 0.45],
#                               [0.50, 0.55]]))
# output1.set_bias(np.array([[0.60],
#                            [0.60]]))

model1 = Sequential()
model1.add(input1, hidden1, output1)
model1.compile(GradientDescent(learning_rate=0.9), loss='mse', metric='mse')
epochs = 20_0
errors = model1.fit(features, labels, epochs=epochs, batch_size=2, verbose=2)
plt.plot(np.arange(epochs), errors)
plt.gca().set_yscale('log')
plt.grid('on')
plt.gca().set_ylim([1e-20, 1e-0])

plt.show()

# print(model1.predict())