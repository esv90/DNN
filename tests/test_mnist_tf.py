import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import itertools

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if False:
    # Plot some of the digits
    fig, axes = plt.subplots(nrows=4, ncols=4)
    for i, j in itertools.product(range(4), range(4)):
        idx = np.random.randint(len(x_train))
        axes[i][j].imshow(x_train[idx], cmap=plt.cm.gray)
        axes[i][j].set_title(y_train[idx])
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])

    plt.tight_layout()
    plt.show()

# Reshape the data to fit the expected shape of keras.Conv2D
# expand_dims take the original array and adds one dimension specified by the axis parameter
# In this case we add one axis at the end of the array shape == (n,28,28) --> shape == (n,28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# Hot-encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(7, 3, activation='relu', padding='valid'))
model.add(Conv2D(3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_split=0.2)