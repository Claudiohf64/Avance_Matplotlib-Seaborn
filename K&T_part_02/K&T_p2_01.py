import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=100, verbose=0)
res = model.predict(np.array([[13.0, 1.78, 2.14, 2.58, 1.04,2.52, 3.0, 0.24, 2.5, 6.5, 0.3, 0.3, 1.0]]))
print(wine.target_names[np.argmax(res,1)])


