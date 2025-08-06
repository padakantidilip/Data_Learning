import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train[0])

#to_categorical
y_train = to_categorical(y_train)
print(y_train[0])

#architecture
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(10,activation="softmax"))


model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
