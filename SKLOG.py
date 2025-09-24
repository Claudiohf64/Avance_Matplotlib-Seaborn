from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([0, 0, 0, 1, 1, 1, 1])

modelo_logistico = LogisticRegression()
modelo_logistico.fit(X, y)

prediccion = modelo_logistico.predict([[3.5]])
print(prediccion)
