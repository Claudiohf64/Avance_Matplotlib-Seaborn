# * Reduccion de la dimensionalidad 
from sklearn.decomposition import PCA #analizis de componentes principales
import numpy as np
from sklearn.datasets import load_iris

iris=load_iris()
x = iris.data
print(x[0:5])

#Indica con cuantas columnas se va a quedar y en n_components se le indica con cuantos componentes se va a trabajar
model=PCA(n_components=2)

iris_reduced=model.fit_transform(x)

datos_indiv=iris_reduced[0:5]

iris_inverse=model.inverse_transform(datos_indiv)

print(iris_inverse)

