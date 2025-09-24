import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data()#dividimos el dataset
print(y_train.shape)
plt.imshow(x_test[0],cmap='gray')#graficamos la primera imagen
plt.show()#mostramos el gráfico 