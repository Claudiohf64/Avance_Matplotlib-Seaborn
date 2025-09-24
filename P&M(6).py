import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
iris=load_iris()
frame= pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
frame['target']=iris.target
frame['target_names']=iris.target_names[iris.target]
# r1=frame.query()
reporte1=frame.groupby('target_names')['target'].count()
# reporte2=reporte1[]

#grafico de barras
plt.bar(reporte1.index,reporte1.values)
plt.show()

# #grafico pastel
# plt.pie(reporte1,labels=reporte1.index, autopct='%1.1f%%')
# plt.show()
# print(frame['target'])

# #grafico de barras(comparando a setosa y virginica)


