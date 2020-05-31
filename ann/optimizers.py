# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:48:39 2020

@author: a339594
"""
import numpy as np
import ann
from ann.layers import Flatten


class SGD:
    """ Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.mean_error = None
        self.sum_error = None

    def fit(self, features, labels, layers):
        """ Fit the features to the labels by tweaking the weights and biases using gradient descent"""
        # First, we make sure that the first layer is the Input layer which we pass the features to
        if not isinstance(layers[0], ann.layers.Input):
            raise TypeError('First layer must be of type Input layer')

        if len(features) != len(labels):
            raise ValueError('The number of features does not correspond to the number of labels')

        # Take one step using gradient descent
        batch_size = len(features)
        layers = self._step(features, labels, layers, batch_size)

        # self.mean_error = np.mean(errors)
        # self.sum_error = np.sum(errors)

        # Return the layers with updated weights and biases to the model
        return layers

    def _step(self, features, labels, layers, batch_size):
        """
        Takes a set of features and labels and moves the features forward through the network.
        Then calculates the error between network output and labels and propagate this error backward
        using back propagation to find the appropriate changes (nablas or partial derivatives in bias
        and weights for every neuron. The changes are then be multiplied with the learning rate and
        added to bias and weights
         """

        # Initialize the partial derivatives of the weights and biases
        nabla_w = [np.zeros(layer.weights.shape) for layer in layers[1:]]  # First layer is the input layer and thus has no weights
        nabla_b = [np.zeros(layer.bias.shape) for layer in layers[1:]]

        # Loop over the mini-batch consisting of the features and labels
        for feature, label in zip(features, labels):
            # First, forward propagate all features
            layers = self._forward_propagate(layers, feature)

            # Second, calculate the errors
            error_derivatives = self._cost_derivative(layers[-1].output_values, label)

            # Third, back propagate the errors, i.e. calculate the partial derivatives of
            # weights and bias for this feature and label
            delta_nabla_w, delta_nabla_b = self._backward_propagate(layers, error_derivatives)

            # Update partial derivatives
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # Update the weights and biases
        layers = self._update_weights(layers, nabla_w, batch_size)
        layers = self._update_biases(layers, nabla_b, batch_size)

        return layers

    def _forward_propagate(self, layers, feature):
        """Propagates one feature through the network and returns the layers afterwards"""

        # Pass the features to the input layer
        layers[0].forward_pass(feature)

        # Perform forward pass for all other layers
        for layer in layers[1:]:
            layer.forward_pass()

        return layers

    def _backward_propagate(self, layers, error_derivatives):
        """ Move backwards through the network using back propagation to find the appropriate
        changes (nablas or partial derivatives in bias and weights for every neuron. The changes
         will then be multiplied with the learning rate and added to bias and weights"""
        nabla_w = [np.zeros_like(layer.weights) for layer in layers[1:]]  # list of lists, Input layer has no weights
        nabla_b = [np.zeros_like(layer.bias) for layer in layers[1:]]     # list of lists, Input layer has no biases

        # First, we calculate the delta and partial derivatives of the bias and weights
        # for the last hidden layer where nabla is the symbol for partial derivative
        last_layer = layers[-1]  # The output layer of the network
        delta = error_derivatives * last_layer.derivatives  # Multiply the cost function derivatives with the last layer derivatives (chain rule)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, last_layer.input_layer.output_values.T)

        # Now, we can loop backwards through the network and apply the chain rule at every step. We start at the penultimate
        # layer which has index len(layers)-2 and we move until the second layer (just before the input layer)
        for i in range(len(layers)-2, 0, -1):

            # Update delta
            delta = np.dot(layers[i+1].weights.T, delta) * layers[i].derivatives

            if not isinstance(layers[i], Flatten):
                nabla_b[i-1] = delta
                nabla_w[i-1] = np.dot(delta, layers[i].input_layer.output_values.T)

        return nabla_w, nabla_b

    def _update_weights(self, layers, nabla_w, batch_size):
        """ Updates the layers' weights using gradient descent W - eta * dCost/dW"""

        # Calculate the updated weights for layers 1 to end (the 0th layer is the input layer and thus does not have any weights)
        eta = self.learning_rate / batch_size
        updated_weights = [layer.weights - eta * nabla for layer, nabla in zip(layers[1:], nabla_w)]

        # Set the updated weights in all layers
        [layer.set_weights(weights) for layer, weights in zip(layers[1:], updated_weights)]

        return layers

    def _update_biases(self, layers, nabla_b, batch_size):
        """ Updates the layers' biases using gradient descent b - eta * dCost/db"""

        # Calculate the updated biases for layers 1 to end (the 0th layer is the input layer and thus does not have any biases)
        eta = self.learning_rate / batch_size
        updated_biases = [layer.bias - eta * nabla for layer, nabla in zip(layers[1:], nabla_b)]

        # Set the updated weights in all layers
        [layer.set_bias(bias) for layer, bias in zip(layers[1:], updated_biases)]

        return layers

    def _cost_derivative(self, predictions, labels):
        """ Return the vector of partial derivatives"""
        return predictions - labels


if __name__ == '__main__':
    print('Class for ANN optimizers')
