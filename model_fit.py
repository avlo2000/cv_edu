import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

n = 300
x = np.linspace(1.0, 2.0, n)
y = np.linspace(1.0, 2.0, n)
z = np.linspace(1.0, 2.0, n)
t = np.log(3 * x) + np.log(y) + 15 * np.log(z) + 15 * np.log(z) * np.exp(3 * x)
X = np.stack((x, y, z), axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X, t, train_size=0.8)

poly = PolynomialFeatures(degree=2)
X_train_tr = poly.fit_transform(X_train)

X_test_tr = poly.fit_transform(X_test)


clf = linear_model.LinearRegression()

clf.fit(X_train_tr, y_train)

y_pred = clf.predict(X_test_tr)
print(f"Error norm {np.linalg.norm(y_pred - y_test)}")
plt.plot(y_pred)
plt.plot(y_test)
plt.show()
