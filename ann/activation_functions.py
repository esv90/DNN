# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:21:23 2020

@author: a339594
"""

import numpy as np


def is_activation_function(function_str):
    return function_str in ['relu', 'sigmoid', 'tanh', 'swish', 'linear', 'binary', 'softmax']


def relu(x):
    f = x * (x > 0)
    df = 1 * (x > 0)
    return f, df


def sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    # df = np.exp(-x) / (1 + np.exp(-x))**2
    df = f * (1 - f)  # arguably faster than the above
    return f, df


def tanh(x):
    f = np.tanh(x)
    # df = 1 / np.cosh(x)**2
    df = 1 - f**2  # arguably faster than the above
    return f, df


def swish(x):
    f = x * sigmoid(x)[0]
    df = np.zeros_like(x)
    return f, df


def linear(x):
    """Linear or identity activation function"""
    f = x
    df = np.ones_like(x)
    return f, df


def binary(x):
    # f = 0 if x < 0 else 1
    f = 1 * (x >= 0)
    df = np.zeros_like(x)
    return f, df


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    f = e_x / e_x.sum(axis=0) 
    
    s = f.reshape(-1, 1)
    df = np.diagflat(s) - np.dot(s, s.T)
    
    return f, np.expand_dims(df.diagonal(), axis=-1)
