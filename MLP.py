import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist



# Carregamento e pré-processamento dos dados
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Redimensionamento e normalização dos dados
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Conversão de rótulos em categorias
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Criação do modelo MLP
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(400, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1)

# Avaliação do modelo no conjunto de teste
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
