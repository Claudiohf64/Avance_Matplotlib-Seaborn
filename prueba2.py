from sklearn.tree import DecisionTreeClassifier
import numpy as np

x=np.array([[18],[10],[5]])
y=np.array([2,1,0])
model=DecisionTreeClassifier()
model.fit(x,y)
print(model.predict(np.array([[3]])))