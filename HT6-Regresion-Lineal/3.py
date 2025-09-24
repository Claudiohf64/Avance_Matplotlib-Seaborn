#PROPROCESAMIENTO DE DATOS
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
# One-hot encoding para variables categóricas y StandardScaler para variables numéricas
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
data = pd.DataFrame(
    {
        "tamanio": [50, 80, 60, 120, 100], #StandardScaler
        "habitaciones": [2, 3, 2, 4, 3], #StandardScaler
        "zona": ["Norte", "Sur", "Este", "Oeste"], #OneHotEncoder
        "antiguedad": [10, 5, 20, 2, 15],   #StandardScaler
        "precio": [200000, 300000, 250000, 400000, 350000] #Target
    }
)
x = data[["tamanio", "habitaciones", "zona", "antiguedad"]]
y = data["precio"]
scaler = StandardScaler()
x_num = scaler.fit_transform(data[["tamanio", "habitaciones", "antiguedad"]])
encoder = OneHotEncoder(drop='first', sparse_output=False)
x_cat = encoder.fit_transform(data[["zona"]])
print(x_num)
print(x_cat)

x_final = np.hstack((x_num, x_cat))
print(x_final)

model = Lasso(alpha=0.1)
model.fit(x_final, y)

y_pred = model.predict(x_final)
print("Predicciones: ", y_pred)
print("Coeficientes: ", model.coef_)
print("Intercepto: ", model.intercept_)
print("R^2 Score: ", r2_score(y, y_pred))
print("Mean Squared Error: ", mean_squared_error(y, y_pred))