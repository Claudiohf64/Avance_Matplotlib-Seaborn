# importamos tensorflow para la construccion de RNA
# importamos numpy para los calculos matematicos con vectores
import tensorflow as tf
import numpy as np
# Generamos datos con numpy linspace haciendo una secuancia de numeros entre positivos y negativos
x = np.linspace(-10, 10, 200).reshape(-1, 1)
# Generamos las etiquetas con una condicional para verificar si el valor es positivo o negativo
y = np.array([1 if i > 0 else 0 for i in x]).reshape(-1, 1)
# Definimos el modelo secuencial con keras
# Indicamos el número de neuronas en cada capa ademas de dar una funcion de activacion relu ademas de dar indicar el numero de entradas para las capas
# Por parte de las etiquetas indicamos que se usara una sola neurona con una funcion de activacion sigmoide que es 1 entre 0
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compilamos el modelo con un optimizador adam y una funcion de perdida binaria para filtrar los resultados ya que es de 1 entre 0
model.compile(optimizer='adam', loss='binary_crossentropy')
# Entrenamos el modelo con 100 epocas y utilizamos verbose 0 para no mostrar el progreso
model.fit(x, y, epochs=100, verbose=0)
# creamos una variable que contenga la prediccion utilizando el modelo y le damos los parametros
y_pred = model.predict(np.array([[-3],[-2],[-20],[50]]))
# Imprimir resultados por medio de un bucle donde primero se redondea la prediccion y luego utilizando la variable que contiene la prediccion donde si la prediccion es mayor a 0 se considera positiva en caso contrario negativa
for i in y_pred.round():
    print('positivo' if i > 0 else 'negativo')