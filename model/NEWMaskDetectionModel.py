import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

# Prevent Module Crash
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setting directory for Deep Learning
base_dir = '../dataset/data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Making Data generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.2,
    horizontal_flip=True
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Generating Data for binary classification
# batch size = 20
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

test_generator = train_datagen.flow_from_directory(train_dir,
                                                   batch_size=20,
                                                   class_mode='binary',
                                                   target_size=(150, 150),
                                                   shuffle=True)

# No test data for learning, so separate the data from train data
test_samples = []
test_labels = []

# get 50 data for test
for i in range(50):
    samples, labels = test_generator.next()
    test_samples.extend(samples)
    test_labels.extend(labels)

test_samples = np.array(test_samples)
test_labels = np.array(test_labels)

# Making CNN model for learning
model = tf.keras.models.Sequential([
    # The first Conv Layer
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second Conv Layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=0.25),

    # The third Conv Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fourth Conv Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=0.25),

    # The Fifth Conv Layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten Layer
    tf.keras.layers.Flatten(),

    # Dense Layer for classifier
    tf.keras.layers.Dense(512, activation='relu'),
    # Output Layer, using sigmoid activation function
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model Path
model_path = './maskmodel.h5'
# Early Stopping Method
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

# Learning Model for Classification
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fitting, Used 4 Core for faster learning.
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=20,
                    verbose=1,
                    callbacks=[checkpoint, early_stopping],
                    shuffle=True,
                    workers=4)

# Acc, loss graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Save the model
model.save('maskmodel.h5')


# Evaluate model's F1 score
def evaluate_model(model, test_samples, test_labels):
    predictions = model.predict(test_samples)
    predictions = np.where(predictions > 0.5, 1, 0)
    f1 = f1_score(test_labels, predictions)
    return f1


# F1 score
f1 = evaluate_model(model, test_samples, test_labels)
print("F1 Score: {:.4f}".format(f1))

# Find the best Threshold for classification
thresholds = np.arange(0.1, 1.0, 0.1)
best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    predictions = model.predict(test_samples)
    predictions = np.where(predictions > threshold, 1, 0)
    f1 = f1_score(test_labels, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Best F1 Score: {:.4f}".format(best_f1))
print("Best Threshold: {:.1f}".format(best_threshold))
