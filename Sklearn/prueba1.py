from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
iris=load_iris()
x=iris.data
datos= pd.DataFrame(x, columns=iris.feature_names)
datos=datos[['sepal length (cm)','sepal width (cm)']]
model=KMeans(n_clusters=3,random_state=42)
model.fit(datos)
centers=model.cluster_centers_
labels=model.labels_
plt.scatter(datos['sepal length (cm)'], datos['sepal width (cm)'],c=labels)
plt.scatter(centers[:,0],centers[:,1],marker='o',color='r')
plt.show()

nump = np.array(datos)
clusters = {}
for punto, etiqueta in zip(nump, labels):
    clusters.setdefault(etiqueta, []).append(punto.tolist())
print('Conteo de clusters\n',clusters)

print('Conteo de la columna 0\n',len(clusters[0]))
print('Conteo de la columna 1\n',len(clusters[1]))
print('Conteo de la columna 2\n',len(clusters[2]))

