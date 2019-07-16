import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

regr = linear_model.LinearRegression()

import pandas as pd
df = pd.read_csv('abalone.data.txt', sep=',',header=None)

df = df.replace('M', 1)
df = df.replace('F', 0)
df = df.replace('I', -1)

data = df.as_matrix()
age = data[:, -1]
gender = data[:, 0]

# (a)
training = data[:, 1:-1]  # except age and gender
regr.fit(training, age)
plt.figure(1)
predict = regr.predict(training)
plt.ion()
plt.xlabel("fitted value")
plt.ylabel("residual")
plt.title("residual plot for predicting ages with measurements ignoring gender")
plt.plot(predict	, age - predict, 'bo')

# (b)
training = data[:, 0:-1]
regr.fit(training, age)
plt.figure(2)
predict = regr.predict(training)
plt.ion()
plt.xlabel("fitted value")
plt.ylabel("residual")
plt.title("residual plot for predicting ages with measurements including gender")
plt.plot(predict, age - predict, 'bo')

# (c)
training = data[:, 1:-1]  # except age and gender
regr.fit(training, np.log(age))
plt.figure(3)
predict = regr.predict(training)
plt.ion()
plt.xlabel("fitted value")
plt.ylabel("residual")
plt.title("residual plot for predicting log ages with measurements ignoring gender")
# plt.plot(predict, np.log(age) - predict, 'yo')
plt.plot(np.exp(predict), age - np.exp(predict), 'yo')

# (d)
training = data[:, 0:-1] 
regr.fit(training, np.log(age))
plt.figure(4)
predict = regr.predict(training)
plt.ion()
plt.title("residual plot for predicting log ages with measurements including gender")
plt.xlabel("fitted value")
plt.ylabel("residual")
plt.plot(np.exp(predict), age - np.exp(predict), 'go')


plt.show()