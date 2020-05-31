from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def is_loss_function(loss_function: str) -> bool:
    """
    Checks if the provided loss function is a valid one
    :param loss_function: string with the name of the loss function
    :return: bool with true if the loss function is valid, false otherwise
    """
    return loss_function in ['mse', 'mae', 'categorical_crossentropy', 'binary_crossentropy']


def mse(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the mean squared error between predictions and labels
    :param predictions: numpy ndarray which contains the predictions
    :param labels: numpy ndarray which contains the labels.
    :return: error: float
    """
    assert predictions.shape == labels.shape, 'Predictions and labels do not have equal shapes'
    return mean_squared_error(predictions.reshape(predictions.shape[0], predictions.shape[1]),
                              labels.reshape(labels.shape[0], labels.shape[1]))


def mae(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the mean absolute error between predictions and labels
    :param predictions: numpy ndarray which contains the predictions
    :param labels: numpy ndarray which contains the labels.
    :return: error: float
    """
    assert predictions.shape == labels.shape, 'Predictions and labels do not have equal shapes'
    return mean_absolute_error(predictions, labels)


def categorical_crossentropy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the categorical cross-entropy loss of the predictions
    :param predictions: numpy ndarray which contains the predictions
    :param labels: numpy ndarray which contains the labels
    :return: cross entropy: float
    """
    assert predictions.shape == labels.shape, 'Predictions and labels do not have equal shapes'
    return -np.sum(labels * np.log(predictions))
