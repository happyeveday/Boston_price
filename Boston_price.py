#Code By Yang

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# get data from url
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# polynomial regression using specific features
x, y = data[:, 4].reshape(-1, 1), target

# try using a cubic polynomial regression model
degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x, y)

# generate more data points to draw a smooth curve
x_range = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
y_range_pred = polyreg.predict(x_range)

# draw a smooth curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data points')
plt.plot(x_range, y_range_pred, color='red', label=f'Degree {degree} Polynomial Fit')
plt.xlabel('Feature 5')
plt.ylabel('Target')
plt.legend()
plt.show()