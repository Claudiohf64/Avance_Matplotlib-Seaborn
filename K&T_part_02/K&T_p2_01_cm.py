# importamos tensorflow para la construccion de RNA profundas
# importamos numpy para los arrays
# importamos load_wine para cargar el dataset del vino
# importamos train_test_split para dividir el dataset en entrenamiento y prueba
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# cargamos el dataset de vinos
# añadimos un una variable x para las características y una variable y para las etiquetas
wine = load_wine()
x = wine.data
y = wine.target

# dividimos el dataset en entrenamiento y prueba usamos 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# modelamos la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(13,)), # capa oculta 1: 10 neuronas (utiliza ReLU)
    tf.keras.layers.Dense(26, activation='relu'), # capa oculta 2: 26 neuronas (utiliza ReLU)
    tf.keras.layers.Dense(3, activation='softmax') # capa de salida regular: 3 neuronas (utiliza softmax)
])

#compilamos el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')# uso el optimizador Adam y la funcion de perdida sparse_categorical_crossentropy ya que es adecuada para problemas de clasificacion multiclase
model.fit(x_train, y_train, epochs=100, verbose=0) # entrenamos el modelo con los datos de x_train y y_train y agregamos 100 epocas para aprendizaje y verbose 0 para no mostrar el progreso
res = model.predict(np.array([[13.0, 1.78, 2.14, 2.58, 1.04,2.52, 3.0, 0.24, 2.5, 6.5, 0.3, 0.3, 1.0]]))# realizamos una prediccion y lo guardamos en un variable
print(wine.target_names[np.argmax(res,1)]) # imprime la clase predicha agregando su target 