import pandas as pd
import matplotlib.pyplot as plt

categorias=pd.DataFrame({
    'id_categoria':[1,2,3],
    'nombre_categoria':['Geseosas','Leches','Jugos']
})
productos=pd.DataFrame({
    'id_producto':[1,2,3,4,5,6],
    'nombre_producto':['KR 2L','Black 3L','Sandy 2L','Gloria 360ml','frugos del valle 500ml','gloria 500ml'],
    'monto':[5.5,6.5,5.5,4.5,4.5,5.5],
    'stock':[100,100,100,100,50,50],
    'id_categoria':[1,1,1,2,3,3]
})

reporte1 = pd.merge(categorias,productos,on='id_categoria',how='inner')
conteo=reporte1.groupby('nombre_categoria')['stock'].sum()
plt.plot(conteo.values,conteo.index,marker='o')
plt.show()