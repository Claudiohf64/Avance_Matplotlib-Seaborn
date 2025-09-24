import tensorflow as tf
import numpy as np
x = np.array([[0,1],[0,0],[1,1],[0,0]])
y = np.array([[0],[0],[1],[0]])
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(1,activation='sigmoid',input_shape=(2,))
])
model.compile(optimizer='sgd',loss='binary_crossentropy')

model.fit(x,y,epochs=100,verbose=0)
y_pred=model.predict(np.array([[1,1],[0,1]]))
for i in y_pred.round():
    print('Positivo' if i > 0 else 'Negativo')



