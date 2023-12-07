import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

img_height = 128
img_width = 128
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer, binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=val_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc}')

train_ds.class_names
# plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int)
    
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 6, i * 2 + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Actual - {train_ds.class_names[labels[i]]}')
        
        plt.subplot(3, 6, i * 2 + 2)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Predicted - {train_ds.class_names[predicted_labels[i][0]]}')

plt.show()

# LINK DATASEET:Please download the dataset from the below url

# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia