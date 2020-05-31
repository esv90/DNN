import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def is_metric(metric: str) -> bool:
    """
    Checks if the provided metric is a valid one
    :param metric: string with the name of the metric
    :return: bool with true if the loss function is valid, false otherwise
    """
    return metric in ['mse', 'mae', 'categorical_accuracy']


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


def categorical_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the categorical accuracy between predictions and labels
    :param predictions: numpy ndarray which contains the predictions
    :param labels: numpy ndarray which contains the labels
    :return: accuracy [0, 1]: float
    """
    assert predictions.shape == labels.shape, 'Predictions and labels do not have equal shapes'
    accuracy = 0
    for prediction, label in zip(predictions, labels):
        argmax_prediction = np.argmax(prediction)
        argmax_label = np.argmax(label)
        if argmax_prediction == argmax_label:
            accuracy += 1
        else:
            accuracy += 0

    return accuracy / len(labels)
