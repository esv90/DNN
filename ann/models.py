# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:15:38 2020

@author: a339594
"""
import numpy as np
import ann.metric_functions as metric_functions
import ann
import ann.loss_functions as loss_functions
# from ann_activation_functions import is_activation_function
from ann.layers import Input, Flatten, Dense, Conv2D, Dropout, MaxPooling2D
import random
from typing import List, Tuple


def create_mini_batches(n_features: int, batch_size: int) -> List[np.ndarray]:
    """ Creates mini batches with batch_size equal to batch_size
        The n:th batch will contain the remaining features
    """
    assert isinstance(n_features, int), 'n_features should be an int'
    assert isinstance(batch_size, int), 'batch_size should be an int'
    assert batch_size <= n_features, 'batch_size cannot be larger than the number of features'

    division = n_features // batch_size  # Will seldom be a natural number
    feature_indices = np.arange(n_features)

    # Shuffle the indices to make the batches random
    random.shuffle(feature_indices)

    # Create a list of mini batches
    mini_batches = [feature_indices[batch_size*i : batch_size*(i+1)] for i in range(division+1)]

    # Remove empty list at the end. This happens if division is a natural number
    mini_batches = [mini_batch for mini_batch in mini_batches if len(mini_batch) != 0]

    return mini_batches


class Sequential:
    def __init__(self):
        self.layers = []
        self.has_input = False
        self.has_output = False
        self.is_compiled = False

        self.optimizer = None
        self.loss = None
        self.metrics = []
        self.n_layers = 0

    def add(self, *layers):
        for layer in layers:
            assert isinstance(layer, (Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout)), 'Layer {}: This is not a compatible layer'.format(layer)
            self._add_layer(layer)
            self.n_layers += 1

    def _add_layer(self, layer):
        # Make the previous layer, the layer's input layer. Raise error if the model does not have an input layer
        # print(type(layer))
        if self.has_input:
            try:
                # Set the input for this layer to the upstream layer in the model layer list
                layer.set_input_layer(self.layers[-1])
            except Exception as e:
                print('Layer could not be added because {}'.format(e))
        else:
            if isinstance(layer, Input):
                self.input = layer
                self.has_input = True
            else:
                raise ValueError('The first layer of a sequential model must be an input layer')

        # Checks are good, then we can add the layer at the end of the list
        self.layers.append(layer)
        self.output = layer  # Always make the last added layer the output
        self.has_output = True

    def summary(self):
        HYPHEN_LENGTH = 50

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        # Create a table of the print statements. Add all columns to the table. Calculate the maximum length of each column. Print the items in table
        # using the column width
        column_text = [['Layer (type)', 'Output Shape', 'Param #']]
        for i, layer in enumerate(self.layers):
            text_line = [layer.name + ' ' + str(layer), str(layer.output_shape), str(layer.params)]
            column_text.append(text_line)

            # Keep track of the number of params
            total_params += layer.params
            trainable_params += layer.params

        # Calculate the column widths
        col_width = max(len(word) for text_line in column_text for word in text_line) + 2 # padding

        # Start printing
        print('Model: "Sequential"')
        print(HYPHEN_LENGTH * '-')

        for i, text_line in enumerate(column_text):
            print("".join(word.ljust(col_width) for word in text_line))
            if i == 0:
                print(HYPHEN_LENGTH * '=')
            elif i < len(column_text)-1:
                print(HYPHEN_LENGTH * '-')

        print(HYPHEN_LENGTH * '=')
        print('Total params: {:,}'.format(total_params))
        print('Trainable params: {:,}'.format(trainable_params))
        print('Non-trainable params: {:,}'.format(non_trainable_params))
        print(HYPHEN_LENGTH * '-')

    def compile(self, optimizer, loss: str, metrics: List[str]):
        """
        Compiles the model, i.e. makes it ready to fit, predict and evaluate
        :param optimizer:
        :param loss:
        :param metrics:
        :return:
        """

        assert isinstance(optimizer, ann.optimizers.SGD), 'optimizer must be SGD'
        self.optimizer = optimizer

        assert loss_functions.is_loss_function(loss), 'The provided loss function is not a valid one'
        self.loss = getattr(loss_functions, loss)

        for metric in metrics:
            assert metric_functions.is_metric(metric), '{} is not a valid metric'.format(metric)
            self.metrics.append(getattr(metric_functions, metric))

        # Make sure the model has inputs and outputs - otherwise we shouldn't bother
        if not (self.has_input and self.has_output):
            raise RuntimeError('The model does not have either input or output')

        assert self.has_input, 'The model does not have inputs'
        assert self.has_output, 'The model does not have outputs'
            
        self.is_compiled = True

    def fit(self, features, labels, epochs=5, batch_size=1, validation_data=None, verbose=0):

        # Check that the model is compiled
        if not self.is_compiled:
            raise RuntimeError('The model must be compiled first')

        # Extract validation data if it exists
        if validation_data:
            features_validation = validation_data[0]
            labels_validation = validation_data[1]
            n_validation = len(features_validation)

        # Decide when to print,
        # verbose==0 --> never print,
        # verbose==1 --> print on ten occasions,
        # verbose==2 --> print on all epochs
        print_every = epochs  # Never print
        if verbose == 1:
            print_every = 10
        elif verbose > 1:
            print_every = 1

        n_training = len(features)
        loss = np.zeros(epochs)
        metric = np.zeros((epochs, len(self.metrics)))

        # Loop over all epochs
        for epoch in range(epochs):
            mini_batches = create_mini_batches(n_training, batch_size)
            for mini_batch in mini_batches:
                self.layers = self.optimizer.fit(features[mini_batch], labels[mini_batch], self.layers)

            loss[epoch], metric[epoch] = self.evaluate(features, labels)

            # Print epoch results if verbose
            if (epoch+1) % print_every == 0:
                self.print_epoch(validation_data, epoch, epochs, loss[epoch], metric[epoch])

            # Save the errors

        return loss

    def evaluate(self, features, labels):
        """

        :param features:
        :param labels:
        :return:
        """
        predicted_labels = self.predict(features, labels.shape)
        loss = self.loss(predicted_labels, labels)

        metric_result = np.zeros((1,len(self.metrics)))
        for i, metric in enumerate(self.metrics):
            metric_result[i] = metric(predicted_labels, labels)

        return loss, metric_result

    def predict(self, features, labels_shape):
        """

        :param features:
        :param labels_shape:
        :return:
        """
        predicted_labels = np.zeros(labels_shape)

        # Predict all features
        for i, feature in enumerate(features):

            # Pass the features to the input layer
            self.layers[0].forward_pass(feature)

            # Perform forward pass for all other layers
            for layer in self.layers[1:]:
                layer.forward_pass()

            # Extract result
            predicted_labels[i] = self.layers[-1].output_values

        return predicted_labels

    def print_epoch(self, validation_data, epoch: int, epochs: int, train_loss, train_metric):


        if validation_data:
            val_loss = self.evaluate(validation_data[0], validation_data[1])
            print("Epoch {}/{}: train_loss = {:2.2f}, val_loss = {}".format(epoch + 1, epochs, train_loss, val_loss))

        else:
            print("Epoch {}/{}: loss: {:2.4g}, metric: {:2.4g}".format(epoch + 1, epochs, train_loss, train_metric[0]))


