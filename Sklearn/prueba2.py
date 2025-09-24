from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
x=np.array([[12],[19],[3],[20],[10],[26]])
model=KMeans(n_clusters=2,random_state=42)
model.fit(x)
centers=model.cluster_centers_
labels=model.labels_

for edades,etiquetas in zip(x,labels):
    print('la persona es ' + 'mayor'+' con una edad de ' if edades >= 18 else 'menor'+' con una edad de ',edades, ' donde su etiqueta es ',etiquetas)

plt.scatter(x[:],labels)
plt.show()