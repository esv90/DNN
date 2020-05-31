import numpy as np
import pandas as pd
import os
import string
import matplotlib.pyplot as plt
import seaborn as sbn
import itertools

from ann.models import Sequential
from ann.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from ann.optimizers import SGD


data = pd.read_csv("../datasets/emnist-letters-train.csv", header=None)

# Now split labels and images from original dataframe
x_data = data.iloc[:, 1:]
y_data = data.iloc[:, 0]

# One hot encoding with get_dummies() and you can compare it with the original labels
y_data = pd.get_dummies(y_data)

# Turn our Dataframes into numpy arrays and delete train and test to save up memory
x_data = x_data.values
y_data = y_data.values
# del data


# For some reason, the EMNIST dataset was rotated and flipped and we need to fix that
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])


x_data = np.apply_along_axis(rotate, 1, x_data)


def plot_data():
    # Plot some of the digits
    fig, axes = plt.subplots(nrows=4, ncols=4)
    for i, j in itertools.product(range(4), range(4)):
        idx = np.random.randint(len(x_data))
        axes[i][j].imshow(x_data[idx].reshape([28, 28]), cmap=plt.cm.gray)
        axes[i][j].set_title(list(string.ascii_lowercase)[np.argmax(y_data[idx])])
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])

    plt.tight_layout()
    plt.show()


if False:
    plot_data()


# Reshape to (n, 28, 28, 1)
x_data = x_data.reshape((x_data.shape[0], 28, 28, 1))
y_data = np.expand_dims(y_data, axis=-1)

# Extract ten percent of the data to speed up training
train_index = np.random.randint(int(0.1*len(x_data)), size=(int(0.1*len(x_data))))

# x_data_train = x_data[train_index]
# y_data_train = y_data[train_index]

x_data_train = x_data
y_data_train = y_data


# Create a model
model = Sequential()
model.add(Input(shape=(28, 28, 1), name='input_1'))
# model.add(Conv2D(filter=7, size=3, stride=1, name='conv2d_1', activation='relu', padding='valid'))
# model.add(Conv2D(filter=3, size=3, stride=1, name='conv2d_2', activation='relu', padding='valid'))
# model.add(Conv2D(filter=3, size=3, stride=1, name='conv2d_3',activation='relu', padding='valid'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(32, name='dense_1', activation='relu'))
# model.add(Dense(256, name='dense_1', activation='relu'))
# model.add(Dense(128, name='dense_1', activation='relu'))
model.add(Dense(26, name='dense_2', activation='softmax'))

# Compile it
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Print summary of the model
model.summary()

# Train it
model.fit(x_data_train, y_data_train, batch_size=32, epochs=1000, verbose=2)


# Evaluate it on the test data
# model.evaluate(x_data, y_data)
