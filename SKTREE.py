from sklearn.tree import DecisionTreeClassifier
import numpy as np

x = np.array([[18], [10], [6]])
y = np.array([0,1,2])

model = DecisionTreeClassifier()
model.fit(x, y)

y_pred = model.predict(np.array([[2]]))
print(y_pred)

