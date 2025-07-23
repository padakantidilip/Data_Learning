#Linear Regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


#data
X = np.linspace(0, 10, 1000)
y = 3*X+7+3*np.random.randn(1000)

#define architecture
model = Sequential()
model.add(Dense(units = 1, input_dim = 1, activation= 'linear'))

#compile
model.compile(optimizer = 'sgd', loss = 'mse', metrics=['mae'])

#train
model.fit(X,y, epochs=50)


model.evaluate(X,y)
result = model.predict(X)


plt.scatter(X, y, label='Original Data')
plt.plot(X, result, color ='red', label ='Predicted Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.savefig('4.png')
plt.show()
print("It is working fine, Dont run me that many times!!  It causes me to use my CPU, GPU and other resourses")