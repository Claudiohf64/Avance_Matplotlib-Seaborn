import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

categorias = pd.DataFrame({
    'id_categoria': [1, 2, 3],
    'nombre_categoria': ['Gaseosas', 'Leches', 'Jugos']
})

productos = pd.DataFrame({
    'id_producto': [1, 2, 3, 4, 5, 6],
    'nombre_producto': ['KR 2L', 'Black 3L', 'Sandy 2L', 'Gloria 360ml', 'Frugos del Valle 500ml', 'Gloria 500ml'],
    'monto': [5.5, 6.5, 5.5, 4.5, 4.5, 5.5],
    'stock': [100, 100, 100, 100, 50, 50],
    'id_categoria': [1, 1, 1, 2, 3, 3]
})

# Une los dataframes
reporte1 = pd.merge(categorias, productos, on='id_categoria', how='inner')
conteo = reporte1.groupby('nombre_categoria')['stock'].sum()

# Gráfico pie con colores personalizados
plt.pie(conteo, labels=conteo.index, autopct='%1.1f%%')
plt.title('Stock total por categoría')

# Carga la imagen
img = mpimg.imread('gato.png')

# Añade la imagen con transparencia dentro del gráfico
# Para un pie chart, el uso de imshow con extent es un poco raro, pero lo dejamos
plt.imshow(img, extent=[-1.5, 1.5, -1.5, 1.5], aspect='auto', alpha=0.3)

plt.show()
