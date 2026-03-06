# Resumo sobre pré-processamento de imagens, segmentação de imagens e detecção/classificação de imagens

<p align="center">
  <img src="https://i0.wp.com/blog.dsacademy.com.br/wp-content/uploads/2018/02/computer-vision.png?resize=768%2C227&ssl=1" width="600">
</p>

## Pré-processamento de imagens

### Introdução

O pré-processamento de imagens é a etapa inicial em pipelines de visão computacional e machine learning. Seu objetivo é melhorar a qualidade dos dados de entrada, removendo ruídos, padronizando dimensões e ajustando características da imagem para facilitar o aprendizado dos modelos.

Imagens brutas podem conter variações de iluminação, resolução ou ruído que dificultam a identificação de padrões. Técnicas de pré-processamento ajudam a tornar os dados mais consistentes e informativos, aumentando o desempenho de algoritmos de classificação, segmentação e detecção.

Entre as principais operações de pré-processamento estão:

* Redimensionamento (resize)
* Normalização de pixel
* Conversão para escala de cinza
* Remoção de ruído
* Equalização de histograma
* Data augmentation

### Aplicação
Essas técnicas são amplamente utilizadas em aplicações como reconhecimento facial, diagnóstico médico por imagem e veículos autônomos.

### Bibliotecas e frameworks

* OpenCV
  * Biblioteca muito utilizada em visão computacional
  * Implementa filtros, transformações e manipulação de imagens
* Pillow (PIL)
  * Biblioteca Python para manipulação simples de imagens
* scikit-image
  * Biblioteca científica focada em processamento de imagens
* TensorFlow / Keras
  * Possui módulos de pré-processamento e data augmentation
* PyTorch (torchvision)
  * Contém ferramentas para transformação de imagens em pipelines de deep learning

### Exemplo usando OpenCV

```python
import cv2

# carregar imagem
imagem = cv2.imread("imagem.jpg")

# converter para escala de cinza
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# reduzir ruido com filtro gaussiano
blur = cv2.GaussianBlur(gray, (5,5), 0)

```
## Segmentação

### Introdução

A segmentação de imagens consiste em dividir uma imagem em regiões ou objetos significativos. O objetivo é separar diferentes partes da imagem para facilitar a análise ou identificação de elementos específicos.

Diferentemente da classificação, que atribui um rótulo à imagem inteira, a segmentação trabalha no nível de pixel, identificando exatamente onde cada objeto está localizado.

Existem **dois tipos** principais:

* Segmentação Semântica
  * Classifica cada pixel em uma categoria
  * Exemplo: identificar estrada, carros e pedestres
    
* Segmentação de Instância
  * Diferencia objetos individuais da mesma classe
  * Exemplo: identificar cada pessoa separadamente em uma imagem

### Aplicações

* Diagnóstico médico (segmentação de tumores)
* Agricultura de precisão
* Veículos autônomos
* Monitoramento ambiental

### Bibliotecas e frameworks

* OpenCV
  * Implementa métodos clássicos de segmentação
* scikit-image
  * Possui algoritmos como watershed e thresholding
* TensorFlow
  * Utilizado para modelos de deep learning como U-Net
* PyTorch
  * Muito utilizado para modelos avançados de segmentação
 
### Exemplo

```python
import cv2

# carregar imagem em escala de cinza
imagem = cv2.imread("imagem.jpg", 0)

# aplicar threshold para separar pixels em dois grupos 
_, segmentada = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY) # acima de 127 eh branco, abaixo eh preto

# salvar imagem segmentada
cv2.imwrite("segmentada.jpg", segmentada)

```

## Detecção e classificação

### Introdução

A detecção e classificação de imagens são tarefas centrais em visão computacional.

Classificação de imagens consiste em atribuir uma categoria à imagem inteira.
Exemplo: identificar se uma imagem contém um gato, cachorro ou carro.

Já a detecção de objetos vai além: ela identifica quais objetos estão presentes e onde eles estão localizados, geralmente usando bounding boxes.

### Aplicações
* Sistemas de vigilância
* Reconhecimento facial
* Diagnóstico médico
* Comércio eletrônico
* Veículos autônomos

### Bibliotecas e frameworks

* TensorFlow / Keras
  * Muito utilizado em projetos de deep learning
* PyTorch
  * Popular em pesquisa e desenvolvimento
* YOLO (You Only Look Once)
  * Modelo muito rápido para detecção de objetos
* Detectron2
  * Framework avançado da Meta para detecção e segmentação
* OpenCV DNN
  * Permite rodar modelos de deep learning para detecção

 ### Exemplo com Tensorflow

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# carregar modelo pré-treinado
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# carregar imagem
img = image.load_img("imagem.jpg", target_size=(224,224))
img_array = image.img_to_array(img)

# preparar entrada
img_array = np.expand_dims(img_array, axis=0)

# fazer previsão
predictions = model.predict(img_array)

print(predictions)

```
