# 🛒 Hackathon Forecast Big Data 2025  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)  

Este repositório contém a solução desenvolvida para o **Desafio Técnico – Hackathon Forecast Big Data 2025**, que tem como objetivo prever a **quantidade semanal de vendas por PDV (Ponto de Venda) e SKU** para apoiar o varejo na **reposição de produtos**.  

## 📌 Desafio  

- Desenvolver um **modelo de previsão de vendas (forecast)**.  
- Prever as vendas para **5 semanas de janeiro/2023**.  
- Base de treino: **histórico de 2022**.  
- Dados disponíveis:  
  - **Transações:** Data, PDV, Produto, Quantidade, Faturamento.  
  - **Cadastro de produtos:** Produto, Categoria, Descrição, + atributos.  
  - **Cadastro de PDVs:** PDV, On/Off Prem, Categoria (c-store, g-store, liquor etc.), Zipcode.  
- Dados de teste (**jan/2023**) não foram disponibilizados.  

## 📂 Estrutura do repositório  

📦 hackathon-forecast-bigdata
 ```bash
 ├── data/
 │ ├── raw/ # Dados brutos
 │ └── processed/ # Dados tratados para modelagem
 ├── notebooks/
 │ ├── 01_eda.ipynb # Análise exploratória (EDA)
 │ ├── 02_preprocessing.ipynb # Limpeza e transformação dos dados
 │ ├── 03_modeling.ipynb # Treinamento e avaliação de modelos
 │ ├── 04_forecast.ipynb # Previsões finais
 ├── src/
 │ ├── data_preprocessing.py # Funções de tratamento de dados
 │ ├── feature_engineering.py # Criação de features
 │ ├── models.py # Implementação e treinamento de modelos
 │ └── utils.py # Funções auxiliares
 ├── requirements.txt # Dependências do projeto
 └── README.md # Documentação
 ```

## ⚙️ Tecnologias utilizadas  

- **Linguagem:** Python 3.9+  
- **Bibliotecas principais:**  
  - Pandas, NumPy (manipulação de dados)  
  - Matplotlib, Seaborn (visualização)  
  - Scikit-learn (modelos base e métricas)  
  - Statsmodels / Prophet / XGBoost / LightGBM (modelagem de séries temporais e regressão)  

## 🚀 Como executar  

1. Clone este repositório:

   ```bash
   git clone https://github.com/seu-usuario/hackathon-forecast-bigdata.git
   cd hackathon-forecast-bigdata
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv/Scripts/activate      # Windows
   ```

3. Instale as dependências

   ```bash
   pip install -r requirements.txt
   ```

4. Execute os notebooks na pasta notebooks/ para reproduzir a solução.

## 📊 Métricas de avaliação

Métricas utilizadas: **RMSE, MAPE e R²**.

O modelo final será avaliado com base na acurácia das previsões para as semanas de janeiro/2023.

## 👥 Equipe

- Rodrigo ...
- Rozana da Malta Martins