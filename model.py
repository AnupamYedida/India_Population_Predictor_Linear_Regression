import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_fwf('india_population_dataset.txt')
x_values = data[['Year']]
y_values = data[['Population']]

regression = linear_model.LinearRegression()
regression.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, regression.predict(x_values))
plt.show()
