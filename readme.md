Sistema de Recomendação por Imagens

Introdução

Este projeto implementa um sistema de recomendação por imagens utilizando um modelo pré-treinado (VGG16). Ele é capaz de sugerir imagens similares a uma imagem de consulta com base em seus recursos visuais, utilizando medidas de similaridade.

Funcionalidades

Extração de recursos visuais: Utiliza a camada totalmente conectada da rede VGG16 para gerar vetores de características.

Cálculo de similaridade: Emprega a similaridade do cosseno para medir a proximidade entre as imagens.

Recomendação de imagens: Retorna as imagens mais similares à imagem de consulta.

Requisitos

Ferramentas Necessárias

Python 3.6 ou superior

TensorFlow/Keras (>= 2.0)

NumPy

Scikit-learn

Instalação das Dependências

Execute o seguinte comando para instalar as dependências:

pip install tensorflow numpy scikit-learn

Estrutura do Projeto

/image_recommendation_project
  - image_recommendation_system.py
  - dataset/
      - images/  # Pasta contendo as imagens do banco de dados
      - query.jpg  # Imagem de consulta
  - README.md

Como Usar

Prepare o Dataset:

Coloque as imagens para o banco de dados na pasta dataset/images.

Coloque a imagem de consulta no arquivo dataset/query.jpg.

Execute o Script:

Inicie o sistema com o comando:

python image_recommendation_system.py

Veja os Resultados:

O script exibirá no terminal as imagens recomendadas, ordenadas pela pontuação de similaridade.

Exemplo de Saída

Se você fornecer uma imagem de consulta chamada query.jpg, a saída poderá ser algo como:

Building feature database...
Feature database built successfully.
Recommending similar images...
Recommended Images:
image1.jpg: Similarity Score = 0.9876
image2.jpg: Similarity Score = 0.9765
image3.jpg: Similarity Score = 0.9654
image4.jpg: Similarity Score = 0.9543
image5.jpg: Similarity Score = 0.9432

Funcionamento Interno

1. Extração de Recursos

A extração de características visuais é realizada pela camada fc1 da rede VGG16 pré-treinada no ImageNet. Cada imagem é reduzida a um vetor de 4096 dimensões representando seus aspectos visuais principais.

2. Base de Dados de Recursos

As características de todas as imagens do banco de dados são armazenadas em um dicionário para acesso rápido durante a recomendação.

3. Cálculo de Similaridade

A similaridade entre os vetores de características é calculada usando a similaridade do cosseno, que mede o ângulo entre dois vetores no espaço vetorial.

Considerações Finais

Este sistema pode ser expandido para diversas aplicações práticas, como:

Recomendação de produtos em e-commerce com base em imagens.

Busca por imagens similares em grandes bases de dados visuais.

Aplicações em design e criatividade.