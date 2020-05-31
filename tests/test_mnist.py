import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import itertools

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


from ann.models import Sequential
from ann.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from ann.optimizers import SGD

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

y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

# Create a model
model = Sequential()
model.add(Input(shape=(28, 28, 1), name='input_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(32, name='dense_1', activation='relu'))
model.add(Dense(10, name='dense_2', activation='softmax'))

# Compile it
model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Print summary of the model
model.summary()

# Train it
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2)
