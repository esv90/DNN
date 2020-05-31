import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

import numpy as np

import matplotlib.pyplot as plt


features = np.array([[0.05, 0.10],
                     [0.05, 0.10],
                     [0.05, 0.10],
                     [0.05, 0.10]])

labels = np.array([[0.01, 0.99],
                   [0.01, 0.99],
                   [0.01, 0.99],
                   [0.01, 0.99]])
#
#
# features = np.array([np.array([[0.05], [0.10]])])
# labels   = np.array([np.array([[0.01], [0.99]])])

input = Input(shape=[2,])
dense1 = Dense(2)
output = Dense(2)

model1 = Sequential()
model1.add(input)
model1.add(dense1)
model1.add(output)
model1.compile(optimizer='Adam', metrics=['mse'], loss='mse')

epochs = 20_000
history = model1.fit(features, labels, verbose=0, batch_size=1, epochs=epochs)


plt.figure()
plt.plot(np.arange(epochs), history.history['mse'])
plt.gca().set_yscale('log')
plt.grid('on')
plt.gca().set_ylim([1e-20, 1e-0])

plt.show()

model1.predict(np.array([[0.05, 0.1]]))
