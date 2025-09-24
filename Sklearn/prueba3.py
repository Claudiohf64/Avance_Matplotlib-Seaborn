from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
wine=load_wine()
x=wine.data
datos= pd.DataFrame(x, columns=wine.feature_names)
datos=datos[['alcohol','malic_acid']]
model=KMeans(n_clusters=3,random_state=42)
model.fit(datos)
centers=model.cluster_centers_
labels=model.labels_
plt.scatter(datos['alcohol'], datos['malic_acid'],c=labels)
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
