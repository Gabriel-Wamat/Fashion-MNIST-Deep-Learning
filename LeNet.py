import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import models, layers, callbacks


# Carregamento e pré-processamento dos dados
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1) / 255.0
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1) / 255.0
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

# Construção do modelo de CNN
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback para salvar o melhor modelo durante o treinamento
checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Treinamento do modelo
history = model_cnn.fit(
    train_x, train_y,
    epochs=10,
    validation_data=(test_x, test_y),
    callbacks=[checkpoint]
)

# Exibindo a acurácia por época
for i, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
    print(f"Época {i+1}, Acurácia: {acc:.4f}, Acurácia de Validação: {val_acc:.4f}")

# Avaliação final do modelo
final_loss, final_accuracy = model_cnn.evaluate(test_x, test_y)
print(f"Acurácia final: {final_accuracy:.4f}")
