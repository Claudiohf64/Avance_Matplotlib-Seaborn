from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd
cancer=load_breast_cancer()
x=cancer.data
datos= pd.DataFrame(x, columns=cancer.feature_names)
datos=datos[['mean radius','mean texture']]
model=KMeans(n_clusters=2,random_state=42)
model.fit(datos)
centers=model.cluster_centers_
labels=model.labels_
plt.scatter(datos['mean radius'], datos['mean texture'],c=labels)
plt.scatter(centers[:,0],centers[:,1],marker='o',color='r')
plt.show()

nump = np.array(datos)
clusters = {}
for punto, etiqueta in zip(nump, labels):
    clusters.setdefault(etiqueta, []).append(punto.tolist())
print('Conteo de clusters\n',clusters)

print('Conteo de la columna 0\n',len(clusters[0]))
print('Conteo de la columna 1\n',len(clusters[1]))

lbl2=pd.Series(labels)
print(lbl2.value_counts())