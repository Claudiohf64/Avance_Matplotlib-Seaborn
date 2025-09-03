import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # === GRAFICO DE BARRAS ===
# etiquetas = ['A', 'B', 'C']
# valores = [100, 200, 100]
# plt.bar(etiquetas, valores, color=['red', 'green', 'blue'])
# plt.title('Gráfico de barras')
# plt.xlabel('Etiquetas')
# plt.ylabel('Valores')
# plt.grid(True)
# plt.show()

# # === HISTOGRAMA ===
# notas=[5,5,6,7,8,8,9,9,9,10,15,15,20]
# plt.hist(notas, bins=10, edgecolor='black')
# plt.title('Histograma de Notas')
# plt.xlabel('Notas')
# plt.ylabel('Frecuencia')
# plt.grid(axis='y')
# plt.show()

# # === DISPERCION ===
# x=[1,2,3,4,5]
# y=[2,4,6,8,10]
# plt.scatter(x, y, color='blue')
# plt.title('Gráfico de Dispersión')
# plt.xlabel('Eje X')
# plt.ylabel('Eje Y')
# plt.grid(True)
# plt.show()

# # === LINEAL ===
# x = np.array([1, 2, 3, 4, 5,6,6,6,7,8,9,10])
# plt.plot(x, marker='o')
# plt.title('Gráfico Lineal')
# plt.xlabel('Eje X')
# plt.ylabel('Eje Y')
# plt.grid(True)
# plt.show()

# # === PASTEL ===
# labels=['A', 'B', 'C']
# valores=[100, 100, 100]
# plt.pie(valores,labels=labels, autopct='%1.1f%%')
# plt.title('Gráfico de Pastel')
# plt.show()
