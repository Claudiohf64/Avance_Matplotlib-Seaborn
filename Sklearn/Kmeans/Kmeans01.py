from sklearn.cluster import KMeans
import numpy as np

x = np.array([[1], [2], [2], [8], [9], [10]])
model = KMeans(n_clusters=3, random_state=42)
model.fit(x)
centers = model.cluster_centers_
labels = model.labels_

for i in labels:
    print('positivo' if i == 1 else ('negativo' if i == 0 else ('otro' if i == 2 else 'ninguno')))

print(model.cluster_centers_)
print(model.labels_)

clusters = {}
for punto, etiqueta in zip(x, labels):
    clusters.setdefault(etiqueta, []).append(punto.tolist())

print(clusters)

for punto, etiqueta in zip(x, labels):
    print(punto, etiqueta)

for punto, etiqueta in zip(x, labels):
    print('bajo' if etiqueta == 0 else ('alto'if etiqueta==1 else 'medio', punto))