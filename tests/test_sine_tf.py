import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# Attempt to fit a sine wave with a neuron net



features = np.arange(0, 4*np.pi, 0.01)
features = features.reshape((features.shape[0], 1, 1))
labels = np.sin(features)


input1 = Input(shape=[1,1])
hidden1 = Dense(32, activation='tanh')
output1 = Dense(1, activation='linear')

model1 = Sequential()
model1.add(input1)
model1.add(hidden1)
model1.add(output1)
epochs = 20_0
model1.compile(optimizer='Adam', metrics=['mse'], loss='mse')

def train_model():
    history = model1.fit(features, labels, verbose=1, batch_size=100, epochs=epochs)

    plt.plot(np.arange(epochs), history.history['mse'])
    plt.gca().set_yscale('log')
    plt.grid('on')
    plt.gca().set_ylim([1e-20, 1e-0])
    
    plt.show()
    
train_model()

x = features
y = model1.predict(x)
plt.figure()
plt.plot(x[:,0,0], y[:,0,0], features[:,0,0], labels[:,0,0])