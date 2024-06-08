# Importação das bibliotecas necessárias para construção da rede
import tensorflow as tf
from tensorflow.keras import layers, models

# Carregamento dos dados do dataset MNIST e divisão em conjuntos de treino e teste
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

# Pré-processamento dos dados:
# - Redimensionamento das imagens para o formato adequado para a CNN (adicionando um canal de cor)
train_x = tf.reshape(train_x, (train_x.shape[0], 28, 28, 1))
test_x = tf.reshape(test_x, (test_x.shape[0], 28, 28, 1))
# - Codificação one-hot das etiquetas (labels) de treino e teste
train_y = tf.one_hot(train_y, 10)
test_y = tf.one_hot(test_y, 10)

# Criação do modelo sequencial usando a API Keras
model_cnn = models.Sequential([
    # Primeira camada convolucional com 6 filtros, kernel de tamanho 5x5 e função de ativação ReLU
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

 # Segunda camada convolucional com 16 filtros    
    layers.Conv2D(16, (5, 5), activation='relu'),     
    layers.MaxPooling2D((2, 2)),
    # Camada de achatamento para transformar a matriz 3D de mapas de características em um vetor 1D
    layers.Flatten(),
    # Primeira camada densa com 120 unidades e função de ativação ReLU
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo definindo 'adam' como otimizador, a entropia cruzada categórica como função de perda
# e a acurácia como métrica de avaliação
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Função para treinar o modelo
def train_nn_model(model, train_data, train_labels, epochs=10, batch_size=128):
    # Treinamento do modelo com dados e etiquetas, incluindo uma divisão para validação
    return model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Função para avaliar o modelo
def evaluate_nn_model(model, test_data, test_labels):
    # Avaliação do modelo nos dados de teste para obter a perda e a acurácia
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

# Execução das funções de treino e avaliação
train_nn_model(model_cnn, train_x, train_y)
evaluate_nn_model(model_cnn, test_x, test_y)
