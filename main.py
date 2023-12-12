import numpy as np  # Mengimport library NumPy untuk manipulasi data numerik
import pandas as pd  # Mengimport library Pandas untuk manipulasi data tabular
import matplotlib.pyplot as plt  # Mengimport library Matplotlib untuk visualisasi data
from skimage import exposure  # Mengimport fungsi equalize_hist dari library skimage untuk histogram equalization

import tensorflow as tf  # Mengimport library TensorFlow untuk machine learning
from tensorflow.keras import layers  # Mengimport modul layers dari TensorFlow Keras untuk pembuatan model
from tensorflow.keras.models import Sequential  # Mengimport modul Sequential dari TensorFlow Keras untuk pembuatan model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Mengimport lapisan-lapisan yang akan digunakan dalam model

# Direktori dataset untuk pelatihan, validasi, dan pengujian
train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

# Dimensi gambar yang diharapkan dan ukuran batch
img_height = 128
img_width = 128
batch_size = 32

# Menggunakan image_dataset_from_directory untuk memuat dataset pelatihan
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',  # Menggunakan skala keabuan karena citra adalah citra X-ray
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Menggunakan image_dataset_from_directory untuk memuat dataset validasi
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Menggunakan image_dataset_from_directory untuk memuat dataset pengujian
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Membangun model sequential CNN untuk klasifikasi biner (normal atau pneumonia)
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

# Mengompilasi model dengan optimizer Adam, loss binary_crossentropy, dan metrik akurasi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model menggunakan dataset pelatihan dan validasi
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Evaluasi model pada dataset pengujian
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc}')

# Mendapatkan nama kelas dari dataset pengujian
class_names = train_ds.class_names

# Membuat subplot untuk menampilkan hasil pengolahan citra dan prediksi
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int)
    
    plt.figure(figsize=(15, 10))
    for i in range(9):
        equalized_image = exposure.equalize_hist(images[i].numpy().squeeze())
        
        # Menampilkan citra asli
        plt.subplot(3, 6, i * 2 + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Actual - {class_names[labels[i]]}')
        
        # Menampilkan citra setelah histogram equalization
        plt.subplot(3, 6, i * 2 + 2)
        plt.imshow(equalized_image, cmap='gray')
        plt.title(f'Predicted - {class_names[predicted_labels[i][0]]}')

plt.show()
