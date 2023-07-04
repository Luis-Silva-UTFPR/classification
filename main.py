import os
import cv2
import math
import random

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

main_folder = 'images/Fruits Classification'
test_folder = main_folder + '/test/'
train_folder = main_folder + '/train/'
valid_folder = main_folder + '/valid/'

fruits = {
    "Apple": 0,
    "Banana": 1,
    "Grape": 2,
    "Mango": 3,
    "Strawberry": 4
}

images = {
    'test': [],
    'train': [],
    'valid': []
}

rotulation = {
    'test': [],
    'train': [],
    'valid': []
}

# Número desejado de imagens a manter
num_imagens_desejado = 500

for fruit in fruits:
    # Lista de todas as imagens na pasta de origem
    pasta_origem = os.path.join(train_folder, fruit)
    imagens = os.listdir(pasta_origem)
    if len(imagens) > num_imagens_desejado:
        random.shuffle(imagens)

        # Selecionar as imagens a serem mantidas
        imagens_mantidas = imagens[:num_imagens_desejado]

        # Percorrer todas as imagens na pasta de origem
        for imagem in imagens:
            caminho_imagem = os.path.join(pasta_origem, imagem)
            # Verificar se a imagem deve ser excluída
            if imagem not in imagens_mantidas:
                # Excluir a imagem
                os.remove(caminho_imagem)


class TrainObjects:
    def __init__(
        self,
        fuit_name: str = None,
        folder_path: str = None
    ):
        self.name = fuit_name
        self.folder_path = folder_path
        self.image_paths: dict = {
            self.name: []
        }
        self.colored_images: list = []
        self.grey_images: list = []
        self.blur_images: list = []
        self.rotulation: list = []

    def _get_image_paths(self):
        paths = []
        image_paths = self.folder_path + self.name
        for archive in os.listdir(image_paths):
            image_path = os.path.join(image_paths, archive)
            paths.append(image_path)
        self.image_paths[self.name] = paths

    def _get_colored_images(self):
        self._get_image_paths()
        for image in self.image_paths[self.name]:
            self.colored_images.append(cv2.imread(image))

    def _get_grey_images(self):
        if not self.colored_images:
            self._get_colored_images()
        for image in self.colored_images:
            self.grey_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def _get_blur_images(self):
        if not self.grey_images:
            self._get_grey_images()
        for image in self.grey_images:
            # aumentado ksize para 5
            image = cv2.medianBlur(image, 5)
            image = cv2.resize(image, [100, 100])
            self.blur_images.append(image.flatten())
            self.rotulation.append(fruits.get(self.name))


class TestObjects(TrainObjects):
    pass


class ValidObjects(TrainObjects):
    pass


for fruit in fruits:
    train = TrainObjects(fruit, train_folder)
    train._get_blur_images()
    test = TestObjects(fruit, test_folder)
    test._get_blur_images()
    valid = ValidObjects(fruit, valid_folder)
    valid._get_blur_images()

    images.get('train').extend(train.blur_images)
    rotulation.get('train').extend(train.rotulation)

    images.get('test').extend(test.blur_images)
    rotulation.get('test').extend(test.rotulation)

    images.get('valid').extend(valid.blur_images)
    rotulation.get('valid').extend(valid.rotulation)

svm = SVC(kernel='rbf', C=1.0)

print("Realizando o treino da rede...")
svm.fit(images.get('train'), rotulation.get('train'))
previsoes = svm.predict(images.get('test'))
relatorio = classification_report(rotulation.get('test'), previsoes)

print("[INFO] - Relatorio Blur: ")
print(relatorio)

# Número de imagens a serem plotadas
num_imagens = len(images['test'])

# Calcular o número de linhas e colunas para o grid de imagens
num_linhas = math.ceil(math.sqrt(num_imagens))
num_colunas = math.ceil(num_imagens / num_linhas)

# Tamanho da figura
fig, axes = plt.subplots(num_linhas, num_colunas, figsize=(12, 12))

# Iterar sobre as imagens e previsões
for i, (imagem, label_predita) in enumerate(zip(images['test'], previsoes)):
    # Redimensionar a imagem para o tamanho original
    imagem = imagem.reshape(100, 100)

    # Calcular a posição da imagem no grid
    linha = i // num_colunas
    coluna = i % num_colunas

    # Plotar a imagem com a label predita
    axes[linha, coluna].imshow(imagem, cmap='gray')
    axes[linha, coluna].set_title(list(
        fruits.keys())[label_predita]
    )
    axes[linha, coluna].axis('off')
# Ajustar o espaçamento entre as imagens
plt.tight_layout()
plt.show()

matriz_confusao = confusion_matrix(rotulation['test'], previsoes)
acuracia = accuracy_score(rotulation['test'], previsoes)
print("Acurácia:", acuracia)

labels = list(fruits.keys())

# Plotar a matriz de confusão como um heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    matriz_confusao,
    annot=True,
    cmap='Blues',
    fmt='d',
    xticklabels=labels,
    yticklabels=labels
)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
