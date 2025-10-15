import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,epochs=10,verbose=0)
y_pred=model.predict(np.expand_dims(x_test[0],0))
print(np.argmax(y_pred))
plt.imshow(x_test[0],cmap='gray')
plt.show()