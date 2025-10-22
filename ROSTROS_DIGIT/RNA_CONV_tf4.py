import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
input_img = layers.Input(shape=(28, 28, 1))

x = layers.Flatten()(input_img)

encoded = layers.Dense(64, activation='relu')(x)
decoded = layers.Dense(28 * 28, activation='relu')(encoded)
decoded = layers.Reshape((28, 28, 1))(decoded)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=3, batch_size=256, verbose=1)

img = x_train[0:1]
decoded_img = autoencoder.predict(img)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(decoded_img[0].reshape(28, 28), cmap='gray')
plt.title("Reconstruida")
plt.axis('off')
plt.show()