import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50

# Carregando o dataset Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adicionando uma dimensão de canal
train_images = np.expand_dims(train_images, -1).astype('float32')
test_images = np.expand_dims(test_images, -1).astype('float32')

# Convertendo imagens de grayscale para RGB
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

# Criando tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Função para formatar imagens e converter etiquetas para one-hot
def format_images(image, label):
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    label = tf.one_hot(label, 10)  # Conversão para one-hot
    return image, label

# Aplicando a função de formatação
train_dataset = train_dataset.map(format_images).batch(32)
test_dataset = test_dataset.map(format_images).batch(32)

# Configurando o modelo
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar a base do modelo

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento e avaliação
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Accuracy:', test_acc)
