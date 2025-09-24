import tensorflow as tf
import numpy as np

x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])

y = np.array([x[:,0]+x[:,1]]).reshape(5,1)
y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear',input_shape=(2,))
])

model.compile(optimizer='sgd',loss='mse')
model.fit(x,y,epochs=100,verbose=0)

print(model.predict(np.array([[10,11]])))

