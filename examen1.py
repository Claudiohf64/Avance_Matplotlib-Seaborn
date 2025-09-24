from sklearn.linear_model import LinearRegression
import numpy as np
x=np.array([[10],[5],[20]])
y=np.array([20,10,40])
model=LinearRegression()
model.fit(x,y)
print(model.predict(np.array([[25]])))
