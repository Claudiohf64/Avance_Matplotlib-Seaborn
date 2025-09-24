import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
x = np.array([[100, 50], [120, 60], [400, 200], [380, 220], [700, 300], [720, 320]]) 
model = KMeans(n_clusters=3,random_state=42)
model.fit(x)
centers=model.cluster_centers_
labels=model.labels_
cluster={}
plt.scatter(x[:,0],x[:,1],c=labels)
plt.scatter(centers[:,0],centers[:,1],marker='x',color='r')
plt.show()