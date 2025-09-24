import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
model=tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation='relu',input_shape=(2,)),
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model.compile(optimizer='adam',loss='binary_crossentropy')