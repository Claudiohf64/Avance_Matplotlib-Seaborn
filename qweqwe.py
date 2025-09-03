import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Crea un gráfico simple
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Ejemplo con imagen")

# Carga la imagen (asegúrate de que la ruta sea correcta)
img = mpimg.imread('gato.png')

# Añade la imagen en el gráfico (coordenadas en fracciones del eje)
# Aquí la imagen se pone en la esquina superior derecha
ax.imshow(img, extent=[3.5, 4.5, 25, 35], aspect='auto')

plt.show()
