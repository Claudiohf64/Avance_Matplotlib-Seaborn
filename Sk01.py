from sklearn.tree import DecisionTreeClassifier #Multiclase
from sklearn.linear_model import LogisticRegression #Binaria
from sklearn.linear_model import LinearRegression #Regresión lineal
import numpy as np

x=np.array([18,10,6])
y=np.array([0 if i < 18 else 1 if i < 10 else 2 for i in x])
model=DecisionTreeClassifier()
model.fit(x.reshape(-1, 1), y)
y_pred=model.predict(np.array([[20],[5],[11]]))
print(y_pred)

