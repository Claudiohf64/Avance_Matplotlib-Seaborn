from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30, 35, 40, 45, 50])

modelo_lineal = LinearRegression()
modelo_lineal.fit(X, y)

prediccion = modelo_lineal.predict([[6]])
print(prediccion)

