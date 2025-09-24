# * Reduccion de la dimensionalidad 
from sklearn.decomposition import PCA #analizis de componentes principales
import numpy as np
x = np.array([[2,2],[3,3],[4,4]])

#Indica con cuantas columnas se va a quedar y en n_components se le indica con cuantos componentes se va a trabajar
model=PCA(n_components=1)

x_reduced=model.inverse_transform(x)
print(model)

