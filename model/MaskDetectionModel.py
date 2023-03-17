# NOT USING NOW
# 멍청한 과거의 내 자신의 전유물

import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
  '../dataset/data/train',
  target_size=(128, 128),
  batch_size=64,
  class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
  '../dataset/data/validation',
  target_size=(128, 128),
  batch_size=64,
  class_mode='binary'
)

model = tf.keras.models.Sequential([
    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPool2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(rate=0.25),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(rate=0.25),

    # The fifth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # Flatten
    tf.keras.layers.Flatten(),
    # Neuron (Hidden layer)
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(128, activation='relu'),

    # 1 Output neuron
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

model.summary()

history = model.fit(train_generator, epochs=15)
test_history = model.fit(validation_generator)

model.save('maskmodel.h5')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
#plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='acc')
#plt.plot(history.history['val_accuracy'], 'k--', label='val_acc')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()



