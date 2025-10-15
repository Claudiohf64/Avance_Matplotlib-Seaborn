import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.datasets import mnist

(ds_train,ds_test),ds_info =tfds.load(
    'mnist',
    shuffle_files=True,
    split=['train[:80%]','train[:80%]'],
    as_supervised=True,
    with_info=True
)

def preprocessing(img,label):
    img = tf.image.resize(img,(28,28))
    img = tf.cast(img,tf.float32)/255.0
    return img,label

train_ds = ds_train.map(preprocessing).shuffle(10000).batch(32).prefetch(1)
test_ds = ds_test.map(preprocessing).batch(32).prefetch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

for img ,label in test_ds.take(1):
  pred = model.predict(img)
  plt.imshow(img[2])
  plt.axis('off')