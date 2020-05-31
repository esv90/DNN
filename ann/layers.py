# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:15:38 2020

@author: a339594
"""
import numpy as np
import skimage.measure

from typing import Tuple
from scipy import signal

from ann import activation_functions
from ann.activation_functions import is_activation_function
import ann


class Neuron:
    """ Creates a single neuron with a bias and n_weights weights"""
    def __init__(self, n_weights: int):
        # Make sure that n_weights is an int
        if isinstance(n_weights, int):
            self.weights = np.random.random(n_weights)  # n_weights weights for every neuron
        else:
            raise TypeError('n_weights must be of type int.')

        self.bias = np.random.random()  # One bias for every neuron


class Input:
    """
    Input layer class
    """
    def __init__(self, shape: Tuple, name: str) -> None:
        """
        Constructor for the Input layer class
        :param shape: tuple with the dimensions of the input features
        :param name: string with the name of the layer
        """
        assert isinstance(shape, tuple), 'shape must be a tuple'
        self.shape = shape

        self.name = name
        self.output_values = np.zeros(shape=self.shape)
        self.output_shape = self.output_values.shape
        self.params = 0  # Input layer does not have any trainable params

    def __str__(self):
        return 'Input'

    def forward_pass(self, features: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass which just means propagates the input features
        :param features: np.ndarray
        :return: features: np.ndarray
        """
        assert isinstance(features, np.ndarray), 'features must be an numpy ndarray'
        assert features.shape == self.shape, 'The shape of these output_values do not correspond with the shape of the layer'
        self.output_values = features

    def backward_pass(self) -> None:
        """
        Nothing is done
        :return: None
        """
        pass


class Flatten:
    """
    Flatten layer class
    """
    def __init__(self, name: str) -> None:
        """
        Constructor for the Flatten layer class
        :param name: string with the name of the layer
        """
        assert isinstance(name, str), 'name should be a string'
        self.name = name

        self.has_input = False
        self.input_layer = None

        self.output_values = None
        self.output_shape = None
        self.derivatives = None
        self.params = 0  # Flatten layer does not train anything, thus it does not have any trainable params

        self.weights = None
        self.n_weights = None
        self.weights_initialized = False
        self.bias = None

    def __str__(self):
        return "Flatten"

    def set_input_layer(self, input_layer):
        """
        Specifies the input layer to this Flatten layer
        :param input_layer: ann.layer
        :return: None
        """
        if not isinstance(input_layer, (Input, Conv2D, MaxPooling2D, Dropout)):
            raise TypeError('''Incompatible input layer. Upstream layer must be
                            either "Input", "Conv2D", "MaxPooling2D", "Dropout".''')
    
        self.input_layer = input_layer
        self.has_input = True

        self.output_values = np.zeros(shape=(self.input_layer.output_values.size, 1))
        self.output_shape = self.output_values.shape

        self.weights = np.ones_like(self.output_values)
        self.n_weights = len(self.weights)
        self.weights_initialized = True
        self.bias = np.zeros_like(self.output_values)

    def forward_pass(self):
        # Flatten and Reshape into correct format with padded (:, 1)
        self.output_values = self.input_layer.output_values.flatten().reshape(self.weights.shape[0], self.weights.shape[1])

        if isinstance(self.input_layer, Input):
            self.derivatives = np.ones_like(self.output_values)
        else:
            self.derivatives = self.input_layer.derivatives.flatten().reshape(self.weights.shape[0], self.weights.shape[1])

        self.output_shape = self.output_values.shape

    def set_weights(self, weights):
        pass

    def set_bias(self, bias):
        pass


class Dense:
    """
    Single dense layer of fully connected neurons
    """

    def __init__(self, neurons: int, name: str, activation='relu'):
        """
        Constructor for the Dense layer
        :param neurons: int - number of connected neurons
        :param name: string - name of the layer
        :param activation: string - name of the activation functions
        """
        assert isinstance(neurons, int), 'neurons should be an int'
        self.neurons = neurons

        assert isinstance(name, str), 'name should be a string'
        self.name = name

        assert is_activation_function(activation), 'activation is not valid'
        self.activation = getattr(activation_functions, activation)  # calls the activation function of a module by using its name

        self.has_input = False
        self.input_layer = None
        self.n_weights = None

        # The weights will be contained in a list of numpy lists,
        # where every numpy list contains the weights for each neuron
        self.weights = np.array([])
        self.weights_initialized = False
        self.bias = np.random.rand(self.neurons, 1) # shape = (self.neurons, ) [b1, b2, ..., b_n-1, b_n] The bias for every neuron is a scalar

        # Initialize the neurons' values
        self.output_values = np.empty_like(0, shape=(self.neurons, 1))  # list of empty lists
        self.derivatives = np.empty_like(0, shape=(self.neurons, 1))
        self.backward_values = np.empty_like(0, shape=(self.neurons, 1))
        self.output_shape = self.output_values.shape

    def __str__(self):
        return 'Dense'

    def _initialize_weights(self):
        """
        Initialize the weights for the dense layer using a normal distribution.
        :return: None
        """
        assert self.has_input, 'In order to set the weights, an input layer must be defined'

        # Initialize the weights for every neuron. The number of weights for each neuron
        # is given by the upstream layer
        y = 1 / np.sqrt(self.n_weights)

        # self.weights = np.random.rand(self.neurons, self.n_weights) * y - y/2
        self.weights = np.random.normal(0, y, (self.neurons, self.n_weights))
        self.weights_initialized = True

        self.params = self.weights.size + self.bias.size

    def set_weights(self, weights: np.ndarray):
        """
        Manually set the weights of all neurons
        :param weights: numpy ndarray with shape=(neurons, weights)
        :return:
        """
        assert self.has_input, 'In order to set the weights, an input layer must be defined'

        if isinstance(weights, np.ndarray):
            if weights.shape == (self.neurons, self.n_weights):
                self.weights = weights
                self.weights_initialized = True
            else:
                raise ValueError('''weights must have shape: ({}, {})'''.format(self.weights.shape))
        else:
            raise TypeError('weights must be a numpy.ndarray')

    def set_bias(self, bias: np.ndarray):
        """
        Manually set the bias for all neurons
        :param bias: numpy ndarray with shape=(neurons,1)
        :return:
        """
        assert isinstance(bias, np.ndarray), 'Bias must be a numpy ndarray'
        self.bias = bias

    def set_input_layer(self, input_layer):
        # if type(input_layer) not in [Dense, Input, Flatten]:
        if not isinstance(input_layer, (Dense, Input, Flatten)):
            raise TypeError('''Incompatible layer. Upstream layer must be either
                             "Dense", "Input", or "Flatten"''')
        if input_layer.output_values.ndim > 2 and (1 in input_layer.output_values.shape):
            raise TypeError('Dense layers can only handle 1-D input. Use a Flatten layer to convert 2-D to 1-D')

        self.input_layer = input_layer  # Sets the upstream layer
        self.has_input = True
        self.n_weights = len(self.input_layer.output_values)

        self._initialize_weights()

    def forward_pass(self):
        # First we have to check if the weights are initialized
        if not self.weights_initialized:
            self._initialize_weights()
        
        # Calculate the dot product and then apply activation function
        self.values = np.dot(self.weights, self.input_layer.output_values) + self.bias
        outputs, derivatives = self.activation(self.values)  # [0] are the activations and [1] are the derivatives
        self.output_values = outputs
        self.derivatives = derivatives


class Conv2D:
    def __init__(self, filter: int, size: int, stride: int, name: str, activation='relu', padding='valid'):

        assert isinstance(filter, int), 'filter should be an int'
        self.filter = filter

        assert isinstance(size, int), 'size should be an int'
        self.size = size

        assert isinstance(stride, int), 'stride should be an int'
        self.stride = stride

        assert isinstance(name, str), 'name should be a string'
        self.name = name

        assert is_activation_function(activation), 'activation is not valid'
        self.activation = getattr(activation_functions, activation)  # calls the activation function of a module by using its name

        assert padding in ['same', 'valid'], 'padding should be either "same" or "valid"'
        self.padding = padding

        self.has_input = False
        self.input_layer = None
        self.n_weights = None
        self.weights_initialized = False

        self.bias = np.random.rand(self.filter, 1)  # shape = (self.filter, ) [b1, b2, ..., b_n-1, b_n] The bias for every filter is a scalar


    def __str__(self):
        return 'Conv2D'

    def set_input_layer(self, input_layer):

        # Make sure the input_layer has the correct type and shape
        assert isinstance(input_layer, (Input, Conv2D, MaxPooling2D)), 'Conv2D layer ({}): Incompatible input layer. Input layer should be (Input, Conv2D, MaxPooling2D)'.format(self.name)
        assert input_layer.output_values.ndim == 3, 'Conv2D layer ({}). Incompatible input layer. Input layer should be a 3D tensor'.format(self.name)
        assert input_layer.output_values.shape[0] == input_layer.output_values.shape[1], 'Conv2D layer ({}). Incompatible input layer. The first two dimensions of the inputs should be equal, for instance shape==(28,28,1)'

        self.input_layer = input_layer  # Sets the upstream layer
        self.has_input = True

        self.input_channels = self.input_layer.output_values.shape[2]  # input_channels might be greater than one if for instance we have a RGB image with one channel for red, green, and blue

        self.n_weights = self.filter * self.size * self.size * self.input_channels

        # Create the tensor which keeps the output values
        dim1 = dim2 = self.input_layer.output_values.shape[0] - self.size + 1  # Always a squared filter
        self.output_shape = (dim1, dim2, self.filter)
        self.output_values = np.zeros(shape=self.output_shape)
        self.values = np.zeros(shape=self.output_shape)
        self.derivatives = np.zeros(shape=self.output_shape)

        self._initialize_weights()


    def _initialize_weights(self):
        """ Initialize the weights of the conv2D filter. This is done by creating a numpy array with:

            weights.shape == (size, size, input_channels, filter),

            where filter is the dimensionality of the filter, size is the length of the filter in each dimension, and input_channels is the number of channels
            in the previous layer. For instance, if the input_layer is an ann.Input with RBG data, then the input_channels is equal to 3 (one for red,
            one for green, and one for blue), whereas if the input layer is an ann.Conv2D with filter equal to 7, then the shape is equal to 7.
            The weights are initialized using a normal distribution with mean 0 and standard deviation equal to 1/sqrt(n_weights), where n_weights is
            the total number of weights given by size * size * shape * filter"""

        assert self.has_input, 'Conv2D layer ({}): In order to initialize the weights, an input layer must have been defined'.format(self.name)

        y = 1 / np.sqrt(self.n_weights)
        self.weights = [np.random.normal(0, y, (self.size, self.size)) for _ in range(self.filter)]

        self.weights = np.zeros(shape=(self.size, self.size, self.input_channels, self.filter))  # 4D tensor
        for i in range(self.filter):
            for j in range(self.input_channels):
                self.weights[:, :, j, i] = np.random.normal(0, y, (self.size, self.size))

        # Do it with a list comprehension
        # np.array([[np.random.normal(0, y, (self.size, self.size)) for _ in range(self.shape)] for _ in range(self.filter)]).reshape(self.size, self.size, self.shape, self.filter)  # Also a 4D tensor, but not as readable?

        self.weights_initialized = True

        self.params = self.weights.size + self.bias.size

    def set_bias(self, bias):
        if isinstance(bias, np.ndarray):
            assert bias.shape == (self.filter, 1), 'bias should be of shape ({}, 1)'.format(self.filter)
            self.bias = bias
        else:
            raise ValueError('Bias must be of type np.ndarray')

    def forward_pass(self):
        assert self.has_input, 'Conv2D layer ({}), does not have a specified input layer'.format(self.name)

        # First we have to check if the weights are initialized
        if not self.weights_initialized:
            self._initialize_weights()

        # Convolve the input 2D array in self.input_layer.output_values with the kernel in self.weights

        # Loop over the filters
        for i in range(self.filter):

            # For every filter, perform a convolutional on the input. If the input has more than one channel, loop over all channels to produce a 2D
            # result for every channel. For every channel, we will have one 2D matrix. We sum these matrices up element-wise.
            values = np.zeros_like(self.values[:, :, i])
            for j in range(self.input_channels):
                values += signal.convolve2d(self.input_layer.output_values[:, :, j], self.weights[::-1, ::-1, j, i], self.padding)[::self.stride, ::self.stride]  # the kernel needs to be reversed by definition

            self.values[:, :, i] = values

            # Finally, we perform the activation function to get our output and derivatives
            outputs, derivatives = self.activation(self.values[:, :, i]) + self.bias[i]
            self.output_values[:, :, i] = outputs
            self.derivatives[:, :, 1] = derivatives


class MaxPooling2D:
    def __init__(self, size: int, stride: int, name: str):
        assert isinstance(size, int), 'size should be an int'
        self.size = size

        assert isinstance(stride, int), 'stride should be an int'
        self.stride = stride

        assert isinstance(name, str), 'name should be a string'
        self.name = name

        self.has_input = False
        self.input_layer = None
        self.output_values = None

        self.params = 0

    def __str__(self):
        return "MaxPooling2D"

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        self.has_input = True

        if isinstance(input_layer, Input):
            self.filter = 1
        else:
            self.filter = input_layer.filter  # Must be a Conv2D layer


    def forward_pass(self):
        assert self.has_input, 'MaxPooling2D layer ({}), does not have a specified input layer'.format(self.name)

        self.values = skimage.measure.block_reduce(self.input_layer.output_values, (self.size, self.size), np.max)


class Dropout:
    def __init__(self, name: str, rate=0):
        assert isinstance(name, str), 'name should be a string'
        self.name = name

        assert isinstance(rate, (float)), 'rate should be a float between 0 and 1'
        assert 0.0 <= rate <= 1.0, 'rate is not valid. 0.0 <= rate <= 1.0'

        self.rate = rate
        self.has_input = False
        self.input_layer = None
        self.output_values = None

        self.params = 0

    def __str__(self):
        return "Dropout"

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        self.has_input = True

    def forward_pass(self):
        assert self.has_input, 'Dropout layer ({}), does not have a specified input layer'.format(self.name)

        filter = np.random.binomial(1, self.rate, size=self.input_layer.output_values.shape)
        self.values = self.input_layer.output_values * filter




