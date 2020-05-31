# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:24:04 2020

@author: a339594
"""

import numpy as np

from ann.layers import Input, Dense
from ann.models import Sequential

from ann.optimizers import GradientDescent

# input1 = ann.Input(2)
# dense1 = ann.Dense(2)
# output1 = ann.Dense(2)
# model1 = ann.Sequential()

# model1.add(input1)
# model1.add(dense1)
# model1.add(output1)

# Matt Mazur's example
input1 = Input(input_shape=(2,))
# input1.set_output_values(np.array([0.05, 0.10]))

hidden1 = Dense(2, activation='sigmoid')
hidden1.set_input_layer(input1)
hidden1.set_weights(np.array([[0.15, 0.20],
                              [0.25, 0.30]]))
hidden1.set_bias(0.35)
# print(hidden1.forward_pass())

output1 = Dense(2, activation='sigmoid')
output1.set_input_layer(hidden1)
output1.set_weights(np.array([[0.40, 0.45],
                              [0.50, 0.55]]))
output1.set_bias(0.60)
# print(output1.forward_pass())

model1 = Sequential()

model1.add(input1, hidden1, output1)

model1.compile(GradientDescent(learning_rate=0.5),
               loss='mse', metric='mse')

model1.fit(np.array([0.05, 0.10]),
           np.array([0.01, 0.99]),
           epochs=1)
