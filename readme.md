
# Classificações de Frutas

O seguinte projeto utiliza de um dataset contendo imagens de diversas frutas e utiliza-se de SVC para realizar a classificação das mesmas com base em treino.


## Autores

- Luis Henrique Macedo da Silva - RA 2543931


## 🔗 Dataset
https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification



## Informações

**Main:** Python 

O projeto irá classificar:

- Banana
- Manga
- Morango
- Maça
- Uva

Para treino, cada classe possui mais de 1900 imagens.
Serão utilizada apenas 500 de cada por motivos performáticos.

Serão plotados, as imagens com as labels preditas.
Além também da matriz de confusão


## Documentação

#### O arquivo deve estar estruturado da seguinte maneira



| Pasta root   | pasta para introduçãs das imagens |
| :---------- | :---------: |
| `classification` | `images` |

Dentro da pasta `images`, será introduzida a pasta `Fruits Classification` obtida no zip do dataset.

Seguidos estes passos, basta apenas estar localizado dentro da pasta root e dar o seguinte comando:

```bash
python3 main.py
```

Ou um comando de igual propósito para que possar iniciar o arquivo principal



## Demonstração

valores obtidos com o kernel 'rbf' foram os seguintes

                precision   recall  f1-score   support 

           0       0.22      0.20      0.21        20
           1       0.30      0.35      0.33        20
           2       0.48      0.50      0.49        20
           3       0.27      0.35      0.30        20
           4       0.42      0.25      0.31        20

    accuracy                           0.33       100
    macro avg      0.34      0.33      0.33       100
    weighted avg   0.34      0.33      0.33       100

Sendo:

 - "Apple": 0,
 - "Banana": 1,
 - "Grape": 2,
 - "Mango": 3,
 - "Strawberry": 4


## Bibliotecas necessárias

- OpenCV2
- Scikitlearn
- Matplotlib
- Seaborn

```bash
pip install seaborn
pip install matplotlib
pip install scikit-learn
pip install opencv-python
```






## Funcionalidades

- Pode-se alterar entre os kernels para verificar suas predições
- Se seguidos os passos de estruturação das pastas, o código irá gerar o modelo de classificação sem a necessidade de demais ajustes
- O código plota as imagens com as labels preditas pelo modelo.
