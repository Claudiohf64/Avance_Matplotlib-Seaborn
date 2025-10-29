#CLASIFICACIÓN MULTICLASE RNA CONV (FER_2013)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ROSTROS_DIGIT/archive(6)/train', image_size =(48,48), color_mode = 'grayscale', batch_size = 32
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ROSTROS_DIGIT/archive(6)/test', image_size =(48,48), color_mode = 'grayscale', batch_size = 32
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ROSTROS_DIGIT/archive(6)/validation', image_size =(48,48), color_mode = 'grayscale', batch_size = 32
)

class_names= train_ds.class_names
print(class_names)


train_ds= train_ds.map(lambda x,y: (x/255.0, y))
test_ds= test_ds.map(lambda x,y: (x/255.0, y))
val_ds= val_ds.map(lambda x,y: (x/255.0, y))

for imgs, labels in train_ds.take(1):
  for i in range(6):
    plt.subplot(2,3,i+1)
    eti=labels[i]
    plt.title(class_names[eti])
    plt.imshow(imgs[i])
    plt.axis('off')
plt.show()

print(class_names)
print(labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48,48,1)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(len(class_names),activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=15, validation_data= val_ds)
for images, labels in test_ds.take(1):
    img = images[5]
    true_label = labels[5].numpy()
    img_array= np.expand_dims(img, 0)
    pred = np.argmax(model.predict(img_array),1)[0]
    plt.imshow(img)
    plt.title(class_names[pred])
    plt.show()

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.grid()
plt.legend()
plt.show()