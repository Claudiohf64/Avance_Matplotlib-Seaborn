import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np

data = pd.DataFrame(
    {
        'nom':['A','B','C','D','E'],
        'len':['JAVA','JS','JAVA','JS','JAVA'],
        'per':['backend','fronted','backend','fronted','fullstack'],
        'exp':[1,3,1,3,5],
        'sueldo':[1000, 3000 , 1000, 3000, 5000]
    }
)

encoder = OneHotEncoder(sparse_output=False)
data_enconder = encoder.fit_transform(data[["len", "per"]])

escaler = StandardScaler()
data_escaler = escaler.fit_transform(data[["exp", "sueldo"]])

data_unida = np.hstack([data_escaler, data_enconder])

model = KMeans(n_clusters=3, random_state=42)
model.fit(data_unida)

centers = model.cluster_centers_
labels = model.labels_

clusters = {}
for punto, etiqueta in zip(data_unida, labels):
    clusters.setdefault(etiqueta, []).append(punto.tolist())
    print('JR' if etiqueta == 0 else('intermedio' if etiqueta == 1 else 'Senior'))

print(labels)

